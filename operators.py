"""
Operators: the App-facing forms. These ASSOCIATE; they never run training, and
they never run eval (the eval is selected from existing runs). All persistence
goes through the training-run framework (``dataset.add_training_run`` etc.).
"""

import fiftyone.core.evaluation as foev
import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types

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


def _resolve_view_arg(ctx, target):
    """Maps a dropdown choice to a ``train_view`` arg for the engine.

    A saved-view choice is passed through as its **name** (a string) so the
    engine captures the saved-view-name breadcrumb; the synthetic targets
    resolve to view objects; "(none)" -> ``None``.
    """
    if target in (None, _NONE):
        return None
    if target == _CURRENT:
        return ctx.view
    if target == _DATASET:
        return ctx.dataset
    return target  # a saved-view name; the engine resolves + records it


def _clear_eval_link(run):
    """Unlinks the run's current evaluation, clearing both ends of the weld."""
    samples = run.samples
    eval_key = run.config.eval_key
    if eval_key and eval_key in samples.list_evaluations():
        info = samples.get_evaluation_info(eval_key)
        if getattr(info.config, "train_key", None) == run.train_key:
            info.config.train_key = None
            foev.EvaluationMethod.update_run_config(samples, eval_key, info.config)
    run.config.eval_key = None
    run.save_config()


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
        train_key = ctx.params["train_key"]
        # RD1: the engine requires a valid identifier; reject early with a
        # readable message instead of a raw ValueError from register_run.
        if not train_key.isidentifier():
            ctx.ops.notify(
                f"'{train_key}' is not a valid name: use letters, digits and "
                "underscores, and don't start with a digit.",
                variant="error",
            )
            return

        train_view = _resolve_view_arg(ctx, ctx.params["view_target"])
        val_view = _resolve_view_arg(ctx, ctx.params.get("val_target"))
        test_view = _resolve_view_arg(ctx, ctx.params.get("test_target"))
        try:
            # The Log door ASSOCIATES: auto_eval=False (never runs an eval);
            # an existing eval is linked via eval_key. gt/pred are stored as
            # metadata only, so pred_field is not collected here.
            ctx.dataset.add_training_run(
                train_key,
                train_view,
                val_view=val_view,
                test_view=test_view,
                gt_field=ctx.params.get("label_field") or None,
                auto_eval=False,
                checkpoint_uri=ctx.params.get("checkpoint_uri") or None,
                project_url=ctx.params.get("project_url") or None,
                eval_key=ctx.params.get("eval_key") or None,
            )
        except ValueError as e:
            ctx.ops.notify(str(e), variant="error")
            return

        ctx.ops.notify(
            f"Logged training run '{train_key}'", variant="success"
        )
        return {"train_key": train_key}


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

        keys = ctx.dataset.list_training_runs()
        if not keys:
            inputs.view("warning", types.Warning(label="No training runs to edit"))
            return types.Property(inputs)

        key_view = types.DropdownView()
        for k in keys:
            key_view.add_choice(k, label=k)
        inputs.enum("train_key", keys, required=True, view=key_view, label="Training run")

        train_key = ctx.params.get("train_key")
        if not train_key or not ctx.dataset.has_training_run(train_key):
            return types.Property(inputs)  # dynamic: fields appear after selection

        c = ctx.dataset.get_training_info(train_key).config
        _label_input(inputs, ctx, c.gt_field)
        _checkpoint_input(inputs, c.checkpoint_uri)
        _project_input(inputs, c.project_url)
        _eval_input(inputs, ctx, c.eval_key)
        return types.Property(inputs)

    def execute(self, ctx):
        train_key = ctx.params["train_key"]
        run = ctx.dataset.load_training_run(train_key)

        # Scalar metadata edits write straight to the config (not the
        # TrainingResults.checkpoint_uri setter, which is locked post-finish).
        run.config.gt_field = ctx.params.get("label_field") or None
        run.config.checkpoint_uri = ctx.params.get("checkpoint_uri") or None
        run.config.project_url = ctx.params.get("project_url") or None
        run.save_config()

        # Eval association goes through the weld (1:1, refuse on collision).
        new_eval = ctx.params.get("eval_key") or None
        try:
            if new_eval is not None:
                run.link_evaluation(new_eval)  # idempotent if unchanged
            elif run.config.eval_key is not None:
                _clear_eval_link(run)
        except ValueError as e:
            ctx.ops.notify(str(e), variant="error")
            return {"train_key": train_key}

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
