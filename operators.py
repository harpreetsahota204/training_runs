"""
Operators: the App-facing forms. These ASSOCIATE; they never run training, and
they never run eval (the eval is selected from existing runs). All heavy lifting
delegates to lineage.py.
"""

import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .lineage import (
    STORE_NAME,
    add_training_run,
    get_training_run,
    list_training_runs,
    update_training_run,
)

_NONE = "__none__"
_CURRENT = "__current__"
_DATASET = "__dataset__"


def _label_fields(ctx):
    """Top-level label fields on the dataset (e.g. detections / classifications)."""
    out = []
    for name, field in ctx.dataset.get_field_schema().items():
        doc_type = getattr(field, "document_type", None)
        if isinstance(doc_type, type) and issubclass(doc_type, fol.Label):
            out.append(name)
    return out


def _view_dropdown(ctx, optional=False):
    """Saved views + the synthetic targets, as a dropdown. When ``optional``,
    a leading "(none)" choice lets the user skip logging this split."""
    view = types.DropdownView()
    if optional:
        view.add_choice(_NONE, label="(none)")
    view.add_choice(_CURRENT, label="Current view")
    view.add_choice(_DATASET, label="Entire dataset")
    saved = ctx.dataset.list_saved_views()
    for name in saved:
        view.add_choice(name, label=name)
    values = ([_NONE] if optional else []) + [_CURRENT, _DATASET, *saved]
    return values, view


def _eval_dropdown(ctx):
    evals = ctx.dataset.list_evaluations()
    view = types.DropdownView()
    for ek in evals:
        view.add_choice(ek, label=ek)
    return evals, view


def _label_input(inputs, ctx, default=None):
    fields = _label_fields(ctx)
    view = types.DropdownView()
    for name in fields:
        view.add_choice(name, label=name)
    inputs.enum(
        "label_field",
        fields,
        required=False,
        default=default,
        view=view,
        label="Label field (optional)",
        description="The ground-truth label field the model trained on",
    )


def _eval_input(inputs, ctx, default=None):
    evals, view = _eval_dropdown(ctx)
    inputs.enum(
        "eval_key",
        evals,
        required=False,
        default=default,
        view=view,
        label="Associated eval run (optional)",
        description="An existing evaluation to associate with this run",
    )


def _checkpoint_input(inputs, default=None):
    inputs.str(
        "checkpoint_uri",
        default=default,
        label="Model / checkpoint URI",
        description="Where the trained weights live, e.g. s3://bucket/best.pt",
    )


def _project_input(inputs, default=None):
    inputs.str(
        "project_url",
        default=default,
        label="Experiment / tracker URL",
        description="Link to W&B / MLflow / a ticket (optional)",
    )


def _resolve_samples(ctx, target):
    """Maps a dropdown choice to (samples, saved_view_name)."""
    if target in (None, _NONE):
        return None, None
    if target == _CURRENT:
        return ctx.view, None
    if target == _DATASET:
        return ctx.dataset, None
    return ctx.dataset.load_saved_view(target), target


def _split_target_input(inputs, ctx, name, label, description):
    """Adds an optional view-target dropdown for a val/test split."""
    values, view = _view_dropdown(ctx, optional=True)
    inputs.enum(
        name,
        values,
        default=_NONE,
        view=view,
        label=label,
        description=description,
    )


class LogTrainingRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="log_training_run",
            label="Log Training Run",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # 1) name of the training run
        inputs.str("train_key", required=True, label="Training run name")

        # 2) training set + label
        values, view = _view_dropdown(ctx)
        inputs.enum(
            "view_target",
            values,
            default=_CURRENT,
            view=view,
            label="Training data",
            description="The view the model trained on",
        )
        _label_input(inputs, ctx)

        # 3) test and validation sets
        _split_target_input(
            inputs, ctx, "test_target", "Test data (optional)",
            "The test split, if you want to record it",
        )
        _split_target_input(
            inputs, ctx, "val_target", "Validation data (optional)",
            "The validation split, if you want to record it",
        )

        # 4) checkpoint / tracker URIs, then the (optional) eval association
        _checkpoint_input(inputs)
        _project_input(inputs)
        _eval_input(inputs, ctx)
        return types.Property(inputs)

    def execute(self, ctx):
        samples, train_svn = _resolve_samples(ctx, ctx.params["view_target"])
        val_samples, val_svn = _resolve_samples(ctx, ctx.params.get("val_target"))
        test_samples, test_svn = _resolve_samples(ctx, ctx.params.get("test_target"))
        record = add_training_run(
            samples,
            ctx.params["train_key"],
            val_view=val_samples,
            test_view=test_samples,
            checkpoint_uri=ctx.params.get("checkpoint_uri") or None,
            project_url=ctx.params.get("project_url") or None,
            eval_key=ctx.params.get("eval_key") or None,
            label_field=ctx.params.get("label_field") or None,
            saved_view_name=train_svn,
            val_saved_view_name=val_svn,
            test_saved_view_name=test_svn,
            store=ctx.store(STORE_NAME),
        )
        ctx.ops.notify(
            f"Logged training run '{record['train_key']}'", variant="success"
        )
        return {"train_key": record["train_key"]}


class EditTrainingRun(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="edit_training_run",
            label="Edit Training Run",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        keys = list_training_runs(ctx.dataset, store=ctx.store(STORE_NAME))
        if not keys:
            inputs.view("warning", types.Warning(label="No training runs to edit"))
            return types.Property(inputs)

        key_view = types.DropdownView()
        for k in keys:
            key_view.add_choice(k, label=k)
        inputs.enum("train_key", keys, required=True, view=key_view, label="Training run")

        train_key = ctx.params.get("train_key")
        if not train_key:
            return types.Property(inputs)  # dynamic: fields appear after selection

        record = get_training_run(ctx.dataset, train_key, store=ctx.store(STORE_NAME)) or {}
        _label_input(inputs, ctx, record.get("label_field"))
        _checkpoint_input(inputs, record.get("checkpoint_uri"))
        _project_input(inputs, record.get("project_url"))
        _eval_input(inputs, ctx, record.get("eval_key"))
        return types.Property(inputs)

    def execute(self, ctx):
        train_key = ctx.params["train_key"]
        update_training_run(
            ctx.dataset,
            train_key,
            store=ctx.store(STORE_NAME),
            checkpoint_uri=ctx.params.get("checkpoint_uri") or None,
            project_url=ctx.params.get("project_url") or None,
            eval_key=ctx.params.get("eval_key") or None,
            label_field=ctx.params.get("label_field") or None,
        )
        ctx.ops.notify(f"Updated training run '{train_key}'", variant="success")
        return {"train_key": train_key}


class OpenTrainingRunsPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_training_runs_panel",
            label="List Training Runs",
        )

    def execute(self, ctx):
        ctx.ops.open_panel("training_runs", is_active=True, layout="horizontal")
