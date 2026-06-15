"""The "Train model" door: fine-tune a model on the run's views and record the
result through the engine surface only.

The form is task-first: pick what you want to train (detection, classification,
segmentation, pose) -- only tasks some label field can actually drive are
offered -- then the label field choices are filtered to the fields whose type +
contents support that task (segmentation needs a semantic ``Segmentation``
field, a ``Detections`` field with instance masks, or a filled ``Polylines``
field; etc.). The framework choices -- Ultralytics YOLO or a HuggingFace
AutoModel -- are then filtered to those that can train that task + label type
(e.g. pose is YOLO-only, HF segmentation needs semantic masks), so the user
can't pick an incompatible combination. The trained model is funnelled through
``init_training_run`` -> ``apply_model`` -> ``finish`` (in
``training_helpers.record_run``) so the produced record is indistinguishable
from a notebook (door-1) run; the engine owns the record.

The family-specific export/fit/apply work lives in ``train_yolo`` and
``train_hf``; the form scaffolding and engine flow are shared via
``training_helpers``.
"""

import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import train_hf, train_yolo
from . import training_helpers as th
from .operators import _dropdown, _valid_identifier

_FRAMEWORK_LABELS = {
    "ultralytics": "Ultralytics YOLO",
    "huggingface": "HuggingFace AutoModel",
}

# Tasks offered, in display order. A task is shown only when some label field
# can actually drive it (see ``_available_tasks``).
_TASKS = ("detection", "classification", "segmentation", "pose")

# Field suffix to count distinct classes on, per label type (for the notice
# shown after the label field is picked).
_COUNT_PATHS = {
    fol.Detections: "detections.label",
    fol.Polylines: "polylines.label",
    fol.Keypoints: "keypoints.label",
    fol.Classification: "label",
}


def _has_instance_masks(ctx, field):
    """True if any detection in ``field`` carries an instance mask (in-memory
    ``mask`` or on-disk ``mask_path``) -- the prerequisite for segmentation."""
    n = ctx.dataset.count(f"{field}.detections.mask")
    n += ctx.dataset.count(f"{field}.detections.mask_path")
    return n > 0


def _has_filled_polylines(ctx, field):
    """True if any polyline in ``field`` is filled (a polygon/region rather
    than an open curve) -- the prerequisite for segmentation."""
    return ctx.dataset.count_values(f"{field}.polylines.filled").get(True, 0) > 0


def _task_label_types(task):
    """The label types that can drive ``task`` across either framework (the
    union of the YOLO and HF tables -- the single source of truth for which
    field types a task accepts)."""
    out = set(train_yolo.TASK_LABEL_TYPES.get(task, ()))
    out.update(train_hf.TASK_LABEL_TYPES.get(task, ()))
    return tuple(out)


def _fields_for_task(ctx, task):
    """Label fields whose type AND contents can drive ``task``.

    Type alone settles every task except segmentation, where a ``Detections``
    field qualifies only with instance masks and a ``Polylines`` field only
    when filled (a semantic ``Segmentation`` field always qualifies).
    """
    fields = th.label_fields_for(ctx, _task_label_types(task))
    if task != "segmentation":
        return fields

    out = []
    for name in fields:
        label_type = th.label_type(ctx, name)
        if label_type is fol.Segmentation:
            out.append(name)
        elif label_type is fol.Detections and _has_instance_masks(ctx, name):
            out.append(name)
        elif label_type is fol.Polylines and _has_filled_polylines(ctx, name):
            out.append(name)
    return out


def _available_tasks(ctx):
    """Tasks (in display order) that at least one label field can drive."""
    return [task for task in _TASKS if _fields_for_task(ctx, task)]


def _valid_frameworks(ctx, task, label_field):
    """Installed frameworks that can train ``task`` on ``label_field``'s type.

    This is the upfront validation seam: only compatible, installed frameworks
    are offered, so the user can't launch a run that would fail at fit time
    (e.g. HF segmentation on a ``Detections`` field, or pose on HF)."""
    label_type = th.label_type(ctx, label_field)
    if label_type is None:
        return []

    out = []
    yolo_types = train_yolo.TASK_LABEL_TYPES.get(task)
    if train_yolo.HAS_ULTRALYTICS and yolo_types and issubclass(label_type, yolo_types):
        out.append("ultralytics")
    hf_types = train_hf.TASK_LABEL_TYPES.get(task)
    if train_hf.HAS_TRANSFORMERS and hf_types and issubclass(label_type, hf_types):
        out.append("huggingface")
    return out


def _class_notice(ctx, inputs, label_field):
    """Show how many classes the chosen label field has (best-effort)."""
    label_type = th.label_type(ctx, label_field)
    suffix = _COUNT_PATHS.get(label_type)
    if suffix is not None:
        n = sum(1 for v in ctx.dataset.distinct(f"{label_field}.{suffix}") if v)
        label = f"Found {n} classes in '{label_field}'"
    elif label_type is fol.Segmentation:
        targets = (ctx.dataset.mask_targets or {}).get(label_field) or ctx.dataset.default_mask_targets
        if not targets:
            return
        label = f"{len(targets)} mask classes in '{label_field}'"
    else:
        return

    inputs.view("classes", types.Notice(label=label))


class TrainModel(foo.Operator):
    """Fine-tune a YOLO or HuggingFace model and record its lineage."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="train_model",
            label="Train Model",
            description="Fine-tune a model on your views and record its lineage",
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

        th.add_train_key(inputs)

        # 1) Task -- only those some label field can actually drive.
        tasks = _available_tasks(ctx)
        if not tasks:
            inputs.view(
                "no_fields",
                types.Warning(
                    label="No trainable label fields found",
                    description=(
                        "Need a Detections, Classification, Keypoints, "
                        "Segmentation, or filled Polylines field"
                    ),
                ),
            )
            return th.page(inputs)
        task_view = types.RadioGroup()
        for t in tasks:
            task_view.add_choice(t, label=t.capitalize())
        inputs.enum(
            "task",
            tasks,
            default=tasks[0],
            required=True,
            view=task_view,
            label="Task",
        )
        task = ctx.params.get("task")
        if task not in tasks:
            task = tasks[0]

        # 2) Label field -- those whose type + contents can drive this task.
        fields = _fields_for_task(ctx, task)
        inputs.enum(
            "label_field",
            fields,
            required=True,
            view=_dropdown(fields),
            label="Label field",
            description="The ground-truth field to train on",
        )
        label_field = ctx.params.get("label_field")
        if label_field not in fields:
            return th.page(inputs)
        _class_notice(ctx, inputs, label_field)

        # 3) Training data (train/val/test).
        th.add_splits(inputs, ctx)

        # 4) Framework -- only those that can train this task + label type.
        frameworks = _valid_frameworks(ctx, task, label_field)
        if not frameworks:
            inputs.view(
                "no_framework",
                types.Error(
                    label="No compatible training framework",
                    description=(
                        "No installed framework can train this task on the "
                        f"'{label_field}' field type"
                    ),
                ),
            )
            return th.page(inputs)
        framework_view = types.RadioGroup()
        for value in frameworks:
            framework_view.add_choice(value, label=_FRAMEWORK_LABELS[value])
        inputs.enum(
            "framework",
            frameworks,
            default=frameworks[0],
            required=True,
            view=framework_view,
            label="Framework",
        )
        framework = ctx.params.get("framework")
        if framework not in frameworks:
            framework = frameworks[0]

        # 5) Model + hyperparameters (framework-specific).
        if framework == "huggingface":
            ready = train_hf.hf_model_inputs(inputs, ctx, task)
            pred_default = "hf_predictions"
        else:
            ready = train_yolo.yolo_model_inputs(inputs, ctx, task)
            pred_default = "yolo_predictions"
        if not ready:
            return th.page(inputs)

        # 6) Predictions field, checkpoint output, experiment tracker.
        th.add_pred_field(inputs, pred_default)
        th.add_output(inputs, ctx)
        th.add_project_url(inputs)
        return th.page(inputs)

    def execute(self, ctx):
        train_key = ctx.params["train_key"]
        if not _valid_identifier(ctx, train_key):
            return

        # Building the spec is cheap (resolves splits + config); the heavy fit
        # runs inside record_run's run block. Guard the whole thing so a
        # StopIteration leaking from the HF Trainer can't poison the async loop.
        framework = ctx.params.get("framework", "ultralytics")
        core = train_hf.train_hf if framework == "huggingface" else train_yolo.train_yolo
        spec = core(ctx, train_key)
        return th.guard_stop_iteration(
            lambda: th.record_run(ctx, train_key, **spec)
        )

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
