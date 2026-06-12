"""
Operators: the App-facing forms. The Log/Edit forms ASSOCIATE (they never run
training or eval); the Evaluate form runs a real FO evaluation through the
engine surface (``run.evaluate()``), which welds it to the run. All
persistence goes through the training-run framework
(``dataset.add_training_run`` etc.).
"""

import json

import fiftyone.core.evaluation as foev
import fiftyone.core.labels as fol
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.training as fout

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


def _dropdown(values):
    """A DropdownView whose choices are the given values, labeled as-is."""
    view = types.DropdownView()
    for v in values:
        view.add_choice(v, label=v)
    return view


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


def _label_input(inputs, ctx, default=None):
    fields = _label_fields(ctx)
    inputs.enum(
        "label_field",
        fields,
        required=False,
        default=default,
        view=_dropdown(fields),
        label="Label field (optional)",
        description="The ground-truth label field the model trained on",
    )


def _eval_input(inputs, ctx, default=None):
    evals = ctx.dataset.list_evaluations()
    inputs.enum(
        "eval_key",
        evals,
        required=False,
        default=default,
        view=_dropdown(evals),
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

        inputs.enum(
            "train_key",
            keys,
            required=True,
            view=_dropdown(keys),
            label="Training run",
        )

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


def _populated_pred_count(dataset, config, pred_field):
    """The number of samples across the run's frozen splits with populated
    predictions in ``pred_field``."""
    ids = []
    for split in ("train", "val", "test"):
        ids += getattr(config, f"{split}_view_ids", None) or []
    ids = list(dict.fromkeys(ids))
    if not ids:
        return 0
    return dataset.select(ids).exists(pred_field).count()


class EvaluateTrainingRun(foo.Operator):
    """Runs a real FO evaluation for a completed run with no linked eval.

    Everything goes through the engine surface (``run.evaluate()``), so the
    weld (forward ``eval_key`` + ``train_key`` back-pointer) comes for free.
    The form is type-aware: the eval method's inputs are chosen by the gt
    field's label type, via the same ``resolve_eval_kind`` the engine uses.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="evaluate_training_run",
            label="Evaluate Training Run",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Eligible: completed runs with no linked eval
        keys = [
            k
            for k in ctx.dataset.list_training_runs(status="completed")
            if ctx.dataset.get_training_info(k).config.eval_key is None
        ]
        if not keys:
            inputs.view(
                "warning",
                types.Warning(
                    label="No completed training runs without an evaluation"
                ),
            )
            return types.Property(inputs)

        inputs.enum(
            "train_key",
            keys,
            required=True,
            view=_dropdown(keys),
            label="Training run",
        )

        train_key = ctx.params.get("train_key")
        if train_key not in keys:
            return types.Property(inputs)  # dynamic: fields appear after selection

        c = ctx.dataset.get_training_info(train_key).config

        # Door-3 runs may lack gt/pred fields; collect whichever are missing
        fields = _label_fields(ctx)
        for param, label in (
            ("gt_field", "Ground truth field"),
            ("pred_field", "Predictions field"),
        ):
            if getattr(c, param) is None:
                inputs.enum(
                    param,
                    fields,
                    required=True,
                    view=_dropdown(fields),
                    label=label,
                    description=f"This run has no recorded {label.lower()}",
                )

        gt_field = c.gt_field or ctx.params.get("gt_field")
        pred_field = c.pred_field or ctx.params.get("pred_field")
        if not gt_field or not pred_field:
            return types.Property(inputs)

        # No predictions on the run's frozen views -> nothing to evaluate
        n_preds = _populated_pred_count(ctx.dataset, c, pred_field)
        if n_preds == 0:
            inputs.view(
                "no_preds",
                types.Error(
                    label=f"No predictions found in '{pred_field}'",
                    description=(
                        "None of this run's frozen samples have predictions "
                        "in that field. Apply your model first, then evaluate."
                    ),
                ),
            )
            return types.Property(inputs)

        try:
            kind = fout.resolve_eval_kind(ctx.dataset, gt_field)
        except ValueError as e:
            inputs.view("bad_kind", types.Error(label=str(e)))
            return types.Property(inputs)

        inputs.view(
            "info",
            types.Notice(
                label=(
                    f"{kind} evaluation: '{pred_field}' vs '{gt_field}' on "
                    f"{n_preds} predicted samples"
                )
            ),
        )

        inputs.str(
            "eval_key",
            default=train_key,
            required=True,
            label="Evaluation key",
            description="Defaults to the training run's name",
        )

        if kind == "detection":
            inputs.float(
                "iou",
                default=0.5,
                label="IoU threshold",
                description="Matching threshold between predicted and GT objects",
            )
            inputs.bool(
                "classwise",
                default=True,
                label="Classwise",
                description="Only match objects with the same class label",
            )
            inputs.bool(
                "compute_mAP",
                default=False,
                label="Compute mAP",
                description=(
                    "Sweep all IoUs to compute mAP (slower; required for "
                    "PR curves)"
                ),
            )
        elif kind == "classification":
            methods = ["simple", "top-k", "binary"]
            inputs.enum(
                "method",
                methods,
                default="simple",
                view=_dropdown(methods),
                label="Method",
                description="top-k requires logits; binary requires two classes",
            )
            method = ctx.params.get("method", "simple")
            if method == "top-k":
                inputs.int("k", default=5, label="k")
            elif method == "binary":
                inputs.str(
                    "classes",
                    required=True,
                    label="Classes (neg,pos)",
                    description="Comma-separated: negative label, positive label",
                )
        # segmentation / regression: the defaults are the sensible prototype
        # surface; anything deeper goes through the advanced kwargs below

        inputs.str(
            "eval_kwargs",
            view=types.CodeView(language="json"),
            label="Advanced kwargs (JSON, optional)",
            description=(
                f"Extra kwargs passed to evaluate_{kind}s(), e.g. "
                '{"use_masks": true}'
            ),
        )
        return types.Property(inputs)

    def _build_eval_kwargs(self, ctx, kind):
        """Assembles the ``evaluate_*`` kwargs from the form params.

        Raises:
            ValueError: if the advanced-kwargs JSON is invalid
        """
        kwargs = {}
        if kind == "detection":
            kwargs["iou"] = ctx.params.get("iou", 0.5)
            kwargs["classwise"] = ctx.params.get("classwise", True)
            kwargs["compute_mAP"] = ctx.params.get("compute_mAP", False)
        elif kind == "classification":
            method = ctx.params.get("method", "simple")
            kwargs["method"] = method
            if method == "top-k":
                kwargs["k"] = ctx.params.get("k", 5)
            elif method == "binary":
                classes = [
                    s.strip()
                    for s in (ctx.params.get("classes") or "").split(",")
                    if s.strip()
                ]
                if len(classes) != 2:
                    raise ValueError(
                        "Binary evaluation requires exactly two classes: "
                        "negative,positive"
                    )
                kwargs["classes"] = classes

        raw = ctx.params.get("eval_kwargs")
        if raw:
            try:
                extra = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid advanced kwargs JSON: {e}")
            if not isinstance(extra, dict):
                raise ValueError("Advanced kwargs must be a JSON object")
            kwargs.update(extra)

        return kwargs

    def execute(self, ctx):
        train_key = ctx.params["train_key"]
        run = ctx.dataset.load_training_run(train_key)
        c = run.config

        # Fill in fields a door-3 run may lack (run-framework write, same
        # pattern as the Edit form)
        changed = False
        for field in ("gt_field", "pred_field"):
            if getattr(c, field) is None and ctx.params.get(field):
                setattr(c, field, ctx.params[field])
                changed = True
        if changed:
            run.save_config()

        eval_key = ctx.params.get("eval_key") or train_key
        if not eval_key.isidentifier():
            ctx.ops.notify(
                f"'{eval_key}' is not a valid evaluation key: use letters, "
                "digits and underscores, and don't start with a digit.",
                variant="error",
            )
            return
        if eval_key in ctx.dataset.list_evaluations():
            ctx.ops.notify(
                f"An evaluation '{eval_key}' already exists on this dataset",
                variant="error",
            )
            return

        try:
            kind = fout.resolve_eval_kind(ctx.dataset, c.gt_field)
            kwargs = self._build_eval_kwargs(ctx, kind)
            # The engine runs the eval on the union of the run's populated
            # splits and welds it (eval_key forward, train_key back)
            run.evaluate(eval_key=eval_key, **kwargs)
        except Exception as e:
            # Boundary: surface eval failures (bad kwargs, missing logits,
            # etc.) as a readable notification instead of an operator crash
            ctx.ops.notify(str(e), variant="error")
            return

        ctx.ops.notify(
            f"Evaluated training run '{train_key}' -> eval '{eval_key}'",
            variant="success",
        )
        return {"train_key": train_key, "eval_key": eval_key}


class OpenTrainingRunsPanel(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="open_training_runs_panel",
            label="List Training Runs",
        )

    def execute(self, ctx):
        ctx.ops.open_panel("training_runs", is_active=True, layout="horizontal")
