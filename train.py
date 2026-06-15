"""The "Train model" door: fine-tune an Ultralytics YOLO model on the run's
views and record the result through the engine surface only.

Greenfield Piece 3. The operator exports the chosen train/val views, fine-tunes
a YOLO model, then funnels everything through
``init_training_run`` -> ``apply_model`` -> ``finish`` so the produced record is
indistinguishable from a notebook (door-1) run. The trainer never writes run
storage, config, or status directly -- the engine owns the record.

Only the export/train/apply internals are YOLO-specific; the view resolution,
checkpoint placement, and engine calls are family-agnostic so a future
HuggingFace door can reuse them.
"""

import importlib.util
import os

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.storage as fos
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.training as fout

from .operators import (
    _CURRENT,
    _dropdown,
    _resolve_view_arg,
    _split_target_input,
    _view_dropdown,
)

# ultralytics is an optional dependency; check without importing the heavy
# package at plugin-load time (the real import happens in _train).
_HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None

# YOLO checkpoints offered per task (nano -> xlarge).
YOLO_MODELS = {
    "detection": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"],
    "classification": ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolov8n-cls.pt"],
    "segmentation": ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolov8n-seg.pt"],
    "pose": ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8n-pose.pt"],
}

# The engine's evaluation cleanly scores detection/classification; pose and
# instance-segmentation are recorded without auto-eval in v1.
_AUTO_EVAL_TASKS = ("detection", "classification")

# Ground-truth label types accepted per task.
_TASK_LABEL_TYPES = {
    "detection": (fol.Detections,),
    "classification": (fol.Classification,),
    "segmentation": (fol.Detections, fol.Polylines),
    "pose": (fol.Keypoints,),
}

# Where to read class names from, per task (first non-empty path wins).
_CLASS_PATHS = {
    "detection": ["{f}.detections.label"],
    "segmentation": ["{f}.detections.label", "{f}.polylines.label"],
    "pose": ["{f}.keypoints.label"],
    "classification": ["{f}.label"],
}

_OPTIMIZERS = ["auto", "SGD", "Adam", "AdamW", "RMSProp"]


def _task_label_fields(ctx, task):
    """Top-level label fields whose type matches ``task``."""
    accepted = _TASK_LABEL_TYPES[task]
    out = []
    for name, field in ctx.dataset.get_field_schema().items():
        doc_type = getattr(field, "document_type", None)
        if isinstance(doc_type, type) and issubclass(doc_type, accepted):
            out.append(name)
    return out


def _classes(view, task, label_field):
    """Sorted, non-null class names for the run's label field."""
    for template in _CLASS_PATHS[task]:
        values = [v for v in view.distinct(template.format(f=label_field)) if v is not None]
        if values:
            return sorted(values)
    return []


class TrainModel(foo.Operator):
    """Fine-tune a YOLO model and record its lineage through the engine."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="train_model",
            label="Train Model",
            description="Fine-tune a YOLO model on your views and record its lineage",
            icon="fitness_center",
            dynamic=True,
            # Let the user choose immediate vs delegated (background); training
            # is expensive, so delegated is the default choice.
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        if not _HAS_ULTRALYTICS:
            inputs.view(
                "error",
                types.Error(
                    label="Ultralytics not installed",
                    description="Install it with: pip install ultralytics",
                ),
            )
            return types.Property(inputs, view=types.View(label="Train Model"))

        inputs.str(
            "train_key",
            required=True,
            label="Training run name",
            description="A valid identifier (letters, digits, underscores)",
        )

        task_view = types.RadioGroup()
        for t in YOLO_MODELS:
            task_view.add_choice(t, label=t.capitalize())
        inputs.enum(
            "task",
            list(YOLO_MODELS),
            default="detection",
            required=True,
            view=task_view,
            label="Task",
        )
        task = ctx.params.get("task", "detection")

        fields = _task_label_fields(ctx, task)
        if not fields:
            inputs.view(
                "no_fields",
                types.Warning(
                    label=f"No {task} label fields found",
                    description=f"This dataset has no field of the type {task} needs",
                ),
            )
            return types.Property(inputs, view=types.View(label="Train Model"))

        inputs.enum(
            "label_field",
            fields,
            required=True,
            view=_dropdown(fields),
            label="Label field",
            description="The ground-truth field to train on",
        )
        label_field = ctx.params.get("label_field")
        if not label_field:
            return types.Property(inputs, view=types.View(label="Train Model"))

        classes = _classes(ctx.dataset, task, label_field)
        inputs.view(
            "classes",
            types.Notice(label=f"Found {len(classes)} classes in '{label_field}'"),
        )

        models = YOLO_MODELS[task]
        inputs.enum(
            "model_name",
            models,
            default=models[0],
            required=True,
            view=_dropdown(models),
            label="YOLO model",
        )

        # Train / val / test views (train required; val/test optional).
        values, view = _view_dropdown(ctx)
        inputs.enum(
            "train_target",
            values,
            default=_CURRENT,
            view=view,
            label="Training data",
            description="The view to fine-tune on",
        )
        _split_target_input(
            inputs, ctx, "val_target", "Validation data (optional)",
            "Used for validation during training",
        )
        _split_target_input(
            inputs, ctx, "test_target", "Test data (optional)",
            "Held out for the post-training evaluation",
        )

        inputs.str(
            "pred_field",
            default="yolo_predictions",
            required=True,
            label="Predictions field",
            description="Where the trained model's predictions are written",
        )

        inputs.int("epochs", default=10, label="Epochs")
        inputs.int("imgsz", default=640, label="Image size")
        inputs.int("batch_size", default=16, label="Batch size")
        inputs.float("learning_rate", default=0.01, label="Learning rate")
        inputs.enum(
            "optimizer",
            _OPTIMIZERS,
            default="auto",
            view=_dropdown(_OPTIMIZERS),
            label="Optimizer",
        )
        inputs.int("patience", default=50, label="Early-stopping patience")
        inputs.float(
            "confidence",
            default=0.25,
            label="Confidence threshold",
            description="Minimum confidence for the predictions written back",
        )

        inputs.file(
            "output_parent_dir",
            required=True,
            label="Output parent directory",
            description="Local or cloud directory to store the checkpoint",
            view=types.FileExplorerView(choose_dir=True, button_label="Choose a directory..."),
        )
        inputs.str(
            "model_dir_name",
            default=ctx.params.get("train_key") or "finetuned_model",
            required=True,
            label="Model folder name",
        )

        inputs.str(
            "project_url",
            label="Experiment / tracker URL",
            description="Link to W&B / MLflow / a ticket (optional)",
        )

        return types.Property(inputs, view=types.View(label="Train Model"))

    def execute(self, ctx):
        train_key = ctx.params["train_key"]
        if not train_key.isidentifier():
            ctx.ops.notify(
                f"'{train_key}' is not a valid name: use letters, digits and "
                "underscores, and don't start with a digit.",
                variant="error",
            )
            return

        # The delegated executor wraps execute() in an asyncio Future, which
        # cannot carry a StopIteration -- one can leak from the trainer's
        # internal DataLoader iteration, so translate it to a real error.
        try:
            return self._train(ctx, train_key)
        except StopIteration as e:
            raise RuntimeError(
                "Training iteration ended unexpectedly (StopIteration crossed "
                "the delegated-execution boundary)."
            ) from e

    def _train(self, ctx, train_key):
        from ultralytics import YOLO

        task = ctx.params.get("task", "detection")
        label_field = ctx.params["label_field"]
        pred_field = ctx.params.get("pred_field") or "yolo_predictions"
        model_name = ctx.params["model_name"]

        # The args (a saved-view name, current view, or dataset) are passed to
        # the engine as-is so it records the name breadcrumb; the export needs
        # the resolved view objects.
        train_arg = _resolve_view_arg(ctx, ctx.params["train_target"])
        val_arg = _resolve_view_arg(ctx, ctx.params.get("val_target"))
        test_arg = _resolve_view_arg(ctx, ctx.params.get("test_target"))
        train_view = fout.resolve_view(ctx.dataset, train_arg)
        val_view = fout.resolve_view(ctx.dataset, val_arg)

        hyperparams = {
            "epochs": ctx.params.get("epochs", 10),
            "imgsz": ctx.params.get("imgsz", 640),
            "batch": ctx.params.get("batch_size", 16),
            "lr0": ctx.params.get("learning_rate", 0.01),
            "optimizer": ctx.params.get("optimizer", "auto"),
            "patience": ctx.params.get("patience", 50),
        }

        export_dir = fos.make_temp_dir()
        data_path = _export(train_view, val_view, task, label_field, export_dir)

        model = YOLO(model_name)
        model.train(
            data=data_path,
            project=os.path.join(export_dir, "runs"),
            name="train",
            exist_ok=True,
            **hyperparams,
        )
        checkpoint_uri = _place_checkpoint(ctx, str(model.trainer.best))

        train_config = {"model": model_name, "task": task, **hyperparams}
        if ctx.user_id:
            train_config["triggered_by"] = ctx.user_id

        run = ctx.dataset.init_training_run(
            train_key,
            train_arg,
            val_view=val_arg,
            test_view=test_arg,
            gt_field=label_field,
            pred_field=pred_field,
            auto_eval=None if task in _AUTO_EVAL_TASKS else False,
            project_url=ctx.params.get("project_url") or None,
            train_config=train_config,
        )
        # `with run` marks the record failed (with traceback) on any error.
        with run:
            model.conf = ctx.params.get("confidence", 0.25)
            samples = run.test_view or run.val_view or run.train_view
            run.apply_model(model, samples=samples)
            run.finish(checkpoint_uri=checkpoint_uri)

        ctx.ops.notify(f"Trained model '{train_key}'", variant="success")
        return {
            "train_key": train_key,
            "task": task,
            "model_name": model_name,
            "checkpoint_uri": checkpoint_uri,
            "eval_key": run.eval_key,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        r = ctx.results or {}
        eval_line = f"| **Evaluation** | `{r['eval_key']}` |\n" if r.get("eval_key") else ""
        outputs.str(
            "summary",
            label="Summary",
            view=types.MarkdownView(),
            default=(
                "### Training run recorded\n\n"
                "| Field | Value |\n"
                "|-------|-------|\n"
                f"| **Run** | `{r.get('train_key', 'N/A')}` |\n"
                f"| **Task** | {r.get('task', 'N/A')} |\n"
                f"| **Model** | `{r.get('model_name', 'N/A')}` |\n"
                f"| **Checkpoint** | `{r.get('checkpoint_uri', 'N/A')}` |\n"
                + eval_line
            ),
        )
        return types.Property(outputs, view=types.View(label="Training Complete"))


def _export(train_view, val_view, task, label_field, export_dir):
    """Export the train (and val) views to the on-disk format YOLO trains on.

    Returns the ``data`` argument for ``model.train`` -- a directory for
    classification, a ``dataset.yaml`` otherwise.
    """
    if task == "classification":
        train_view.export(
            export_dir=os.path.join(export_dir, "train"),
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field=label_field,
        )
        if val_view is not None:
            val_view.export(
                export_dir=os.path.join(export_dir, "val"),
                dataset_type=fo.types.ImageClassificationDirectoryTree,
                label_field=label_field,
            )
        return export_dir

    classes = _classes(train_view, task, label_field)
    train_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=classes,
        split="train",
    )
    if val_view is not None:
        val_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split="val",
        )
    return os.path.join(export_dir, "dataset.yaml")


def _place_checkpoint(ctx, best_path):
    """Copy the trained ``best.pt`` into the user's output directory (local or
    cloud) and return its final URI."""
    raw = ctx.params.get("output_parent_dir") or "."
    parent = raw["absolute_path"] if isinstance(raw, dict) else raw
    folder = ctx.params.get("model_dir_name") or "finetuned_model"

    if fos.is_local(parent):
        dest_dir = os.path.join(fos.normalize_path(parent), folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "best.pt")
    else:
        dest = "/".join([parent.rstrip("/"), folder, "best.pt"])

    fos.copy_file(best_path, dest)
    return dest
