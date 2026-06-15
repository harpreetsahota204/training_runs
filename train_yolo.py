"""Ultralytics YOLO training core for the Train Model operator.

Owns the YOLO-specific work: the per-task model lists, the on-disk export,
and ``train_yolo`` (fit + apply-back spec). The form fields, view resolution,
checkpoint placement, and engine calls come from ``training_helpers``.
"""

import importlib.util
import os

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.storage as fos
import fiftyone.utils.training as fout

from .operators import _dropdown
from . import training_helpers as th

# ultralytics is an optional dependency; check without importing the heavy
# package at plugin-load time (the real import happens in train_yolo()).
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None

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

# Ground-truth label types accepted per task (read by the form to filter the
# label field + the framework choices).
TASK_LABEL_TYPES = {
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


def _classes(view, task, label_field):
    """Sorted, non-null class names for the run's label field."""
    for template in _CLASS_PATHS[task]:
        values = [v for v in view.distinct(template.format(f=label_field)) if v is not None]
        if values:
            return sorted(values)
    return []


def yolo_model_inputs(inputs, ctx, task):
    """Add the YOLO model + hyperparameter fields for the chosen ``task``
    (task / label field / splits are collected by the operator)."""
    models = YOLO_MODELS[task]
    inputs.enum(
        "model_name",
        models,
        default=models[0],
        required=True,
        view=_dropdown(models),
        label="YOLO model",
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
    return True


def train_yolo(ctx, train_key):
    """Build the ``record_run`` spec for a YOLO run. Cheap metadata is resolved
    now; the heavy export + fit runs in ``fit`` (inside the engine's run block)."""
    task = ctx.params.get("task", "detection")
    label_field = ctx.params["label_field"]
    model_name = ctx.params["model_name"]
    train_arg, val_arg, test_arg = th.resolve_split_args(ctx)

    hyperparams = {
        "epochs": ctx.params.get("epochs", 10),
        "imgsz": ctx.params.get("imgsz", 640),
        "batch": ctx.params.get("batch_size", 16),
        "lr0": ctx.params.get("learning_rate", 0.01),
        "optimizer": ctx.params.get("optimizer", "auto"),
        "patience": ctx.params.get("patience", 50),
    }

    train_config = {"model": model_name, "task": task, **hyperparams}
    if ctx.user_id:
        train_config["triggered_by"] = ctx.user_id

    # A detections field used for segmentation must be exported as polygons
    # traced from its instance masks (use_masks); polylines export as polygons
    # natively, and detection stays boxes.
    use_masks = task == "segmentation" and th.label_type(ctx, label_field) is fol.Detections

    def fit():
        from ultralytics import YOLO

        train_view = fout.resolve_view(ctx.dataset, train_arg)
        val_view = fout.resolve_view(ctx.dataset, val_arg)
        export_dir = fos.make_temp_dir()
        data_path = _export(
            train_view, val_view, task, label_field, export_dir, use_masks=use_masks
        )

        model = YOLO(model_name)
        model.train(
            data=data_path,
            project=os.path.join(export_dir, "runs"),
            name="train",
            exist_ok=True,
            **hyperparams,
        )
        checkpoint_uri = th.place_checkpoint_file(ctx, str(model.trainer.best))
        model.conf = ctx.params.get("confidence", 0.25)
        return model, checkpoint_uri, None

    return dict(
        train_arg=train_arg,
        val_arg=val_arg,
        test_arg=test_arg,
        gt_field=label_field,
        pred_field=ctx.params.get("pred_field") or "yolo_predictions",
        auto_eval=None if task in _AUTO_EVAL_TASKS else False,
        project_url=ctx.params.get("project_url"),
        train_config=train_config,
        fit=fit,
    )


def _export(train_view, val_view, task, label_field, export_dir, use_masks=False):
    """Export the train (and val) views to the on-disk format YOLO trains on.

    ``use_masks`` exports a detections field as polygons traced from its
    instance masks (for segmentation). Returns the ``data`` argument for
    ``model.train`` -- a directory for classification, a ``dataset.yaml``
    otherwise.
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
    export_kwargs = dict(
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=classes,
    )
    if use_masks:
        export_kwargs["use_masks"] = True
        export_kwargs["tolerance"] = 2

    train_view.export(export_dir=export_dir, split="train", **export_kwargs)
    if val_view is not None:
        val_view.export(export_dir=export_dir, split="val", **export_kwargs)
    return os.path.join(export_dir, "dataset.yaml")
