"""Family-agnostic scaffolding shared by the Train Model doors.

The Ultralytics YOLO core (``train_yolo``) and the HuggingFace core
(``train_hf``) own the family-specific work: exporting / fitting and producing
a model object to apply back. Everything they have in common -- the form
fields, view resolution, checkpoint placement, and the single engine-surface
recording flow (``init_training_run`` -> ``apply_model`` -> ``finish``) -- lives
here so the produced record is identical regardless of which door ran it.
"""

import os

import fiftyone.core.storage as fos
import fiftyone.operators.types as types

from .operators import (
    _CURRENT,
    _resolve_view_arg,
    _split_target_input,
    _view_dropdown,
)

_TITLE = "Train Model"


def page(inputs):
    """Wrap the form ``inputs`` in the operator's titled property."""
    return types.Property(inputs, view=types.View(label=_TITLE))


def label_type(ctx, field):
    """The :class:`fiftyone.core.labels.Label` subclass of ``field``, or None."""
    schema_field = ctx.dataset.get_field_schema().get(field)
    doc_type = getattr(schema_field, "document_type", None)
    return doc_type if isinstance(doc_type, type) else None


def add_train_key(inputs):
    inputs.str(
        "train_key",
        required=True,
        label="Training run name",
        description="A valid identifier (letters, digits, underscores)",
    )


def add_splits(inputs, ctx):
    """Add the train (required) / val / test (optional) view-target dropdowns."""
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


def add_pred_field(inputs, default):
    inputs.str(
        "pred_field",
        default=default,
        required=True,
        label="Predictions field",
        description="Where the trained model's predictions are written",
    )


def add_output(inputs, ctx):
    """Add the output directory + model folder name (local or cloud)."""
    inputs.file(
        "output_parent_dir",
        required=True,
        label="Output parent directory",
        description="Local or cloud directory to store the checkpoint",
        view=types.FileExplorerView(
            choose_dir=True, button_label="Choose a directory..."
        ),
    )
    inputs.str(
        "model_dir_name",
        default=ctx.params.get("train_key") or "finetuned_model",
        required=True,
        label="Model folder name",
    )


def add_project_url(inputs):
    inputs.str(
        "project_url",
        label="Experiment / tracker URL",
        description="Link to W&B / MLflow / a ticket (optional)",
    )


def resolve_split_args(ctx):
    """The train/val/test args passed verbatim to the engine.

    Saved-view choices pass through as their name (a string) so the engine
    records the breadcrumb; tags and the synthetic targets resolve to views.
    """
    return (
        _resolve_view_arg(ctx, ctx.params["train_target"]),
        _resolve_view_arg(ctx, ctx.params.get("val_target")),
        _resolve_view_arg(ctx, ctx.params.get("test_target")),
    )


def _output_dir(ctx):
    """The user's chosen destination ``<parent>/<model_dir_name>`` (local
    normalized path or cloud URI). ``output_parent_dir`` is a dict from the
    App's FileExplorerView and a plain string from SDK callers."""
    raw = ctx.params.get("output_parent_dir") or "."
    parent = raw["absolute_path"] if isinstance(raw, dict) else raw
    folder = ctx.params.get("model_dir_name") or "finetuned_model"
    if fos.is_local(parent):
        return os.path.join(fos.normalize_path(parent), folder)
    return parent.rstrip("/") + "/" + folder


def place_checkpoint_file(ctx, src_path, filename="best.pt"):
    """Copy a single checkpoint file into the output directory; return its URI."""
    dest_dir = _output_dir(ctx)
    if fos.is_local(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
    else:
        dest = dest_dir + "/" + filename
    fos.copy_file(src_path, dest)
    return dest


def place_checkpoint_dir(ctx, src_dir):
    """Copy a checkpoint directory into the output directory; return its URI."""
    dest_dir = _output_dir(ctx)
    fos.copy_dir(src_dir, dest_dir)
    return dest_dir


def record_run(
    ctx,
    train_key,
    train_arg,
    val_arg,
    test_arg,
    gt_field,
    pred_field,
    auto_eval,
    project_url,
    train_config,
    fit,
):
    """Register the run, train, and finish through the engine surface.

    The run is registered (status ``in_progress``, views frozen) BEFORE training
    so the panel shows it as running; then ``fit()`` trains and returns
    ``(model, checkpoint_uri, apply_kwargs)``, the model is applied to the eval
    split, and ``finish()`` runs auto-eval + welds it. Everything after ``init``
    runs in the ``with run`` block, so a failure (training, apply, or eval) is
    recorded on the run (status ``failed`` + full traceback) and summarized in
    the note.
    """
    run = ctx.dataset.init_training_run(
        train_key,
        train_arg,
        val_view=val_arg,
        test_view=test_arg,
        gt_field=gt_field,
        pred_field=pred_field,
        auto_eval=auto_eval,
        project_url=project_url or None,
        train_config=train_config,
    )
    try:
        with run:
            model, checkpoint_uri, apply_kwargs = fit()
            samples = run.test_view or run.val_view or run.train_view
            run.apply_model(model, samples=samples, **(apply_kwargs or {}))
            run.finish(checkpoint_uri=checkpoint_uri)
    except Exception as e:
        _note_failure(run, e)
        raise

    ctx.ops.notify(f"Trained model '{train_key}'", variant="success")
    return {
        "train_key": train_key,
        "model_name": train_config.get("model"),
        "task": train_config.get("task"),
        "checkpoint_uri": run.checkpoint_uri,
        "eval_key": run.eval_key,
    }


def _note_failure(run, exc):
    """Stamp a one-line failure summary into the run's note. The full traceback
    is already stored in ``config.error`` by the run's failure handler; the note
    surfaces the cause in the panel without the user digging into the run."""
    try:
        run.config.note = f"Training failed - {type(exc).__name__}: {exc}"
        run.save_config()
    except Exception:
        pass


def guard_stop_iteration(fn):
    """Run ``fn()``, translating a leaked ``StopIteration`` into a RuntimeError.

    The delegated executor wraps ``execute()`` in an asyncio Future, which
    cannot carry a ``StopIteration`` -- one can leak from a trainer's internal
    DataLoader iteration at epoch boundaries.
    """
    try:
        return fn()
    except StopIteration as e:
        raise RuntimeError(
            "Training iteration ended unexpectedly (StopIteration crossed "
            "the delegated-execution boundary)."
        ) from e
