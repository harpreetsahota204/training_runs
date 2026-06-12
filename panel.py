"""
The Training Runs panel.

A Python ``Panel`` that owns the data + actions and delegates rendering to a
React component (``TrainingRunsView``) via ``composite_view=True``. Panel
methods are wired straight to the frontend -- no standalone operators -- and
data is pushed to React through ``ctx.panel.set_data``. After the Log/Edit
forms succeed, ``on_success`` re-pushes the rows so the panel refreshes itself.
"""

import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.utils.training as fout

_PLUGIN = "@voxel51/training-runs"
_EVAL_PANEL = "model_evaluation_panel_builtin"

# The three splits, in display order.
SPLITS = ("train", "val", "test")


def _split_info(config, split):
    """The per-split ``SplitInfo`` the React panel consumes.

    Presence keys off ``*_view_ids`` (the fact -- a frozen split is present
    iff it has an ID snapshot), NOT the saved-view name (often ``None`` for
    tag/ad-hoc views). The label is the saved-view name if one was passed,
    else "ad-hoc view" when stages exist, else ``None``.
    """
    ids = getattr(config, f"{split}_view_ids", None)
    name = getattr(config, f"{split}_view_name", None)
    stages = getattr(config, f"{split}_view_stages", None)
    return {
        "present": ids is not None,
        "label": name or ("ad-hoc view" if stages else None),
        "num_samples": len(ids or []),
    }


def _run_row(key, info):
    """Adapts a :class:`TrainingInfo` into the ``Row`` shape the React panel
    consumes. The single seam between the run framework and the panel: every
    other handler maps 1:1 to an SDK call.

    Args:
        key: the ``train_key``
        info: the :class:`fiftyone.core.training.TrainingInfo`

    Returns:
        a JSON-safe ``Row`` dict
    """
    c = info.config
    timestamp = getattr(info, "timestamp", None)
    return {
        "train_key": key,
        "checkpoint_uri": c.checkpoint_uri,
        "project_url": c.project_url,
        "eval_key": c.eval_key,
        # the panel's "Label field" is the run's ground-truth field
        "label_field": c.gt_field,
        # Row types created_at as string|null -> serialize in the adapter
        "created_at": timestamp.isoformat() if timestamp else None,
        # review pill, NOT execution status (config.status)
        "status": c.review_status or "new",
        # execution lifecycle: declared/running/completed/failed (+ the
        # scheduled/queued states the delegated path will add later)
        "exec_status": c.status or "declared",
        # opaque per-run training params / script (rendered read-only)
        "train_config": c.train_config or None,
        "note": c.note or "",
        "splits": {s: _split_info(c, s) for s in SPLITS},
    }


def _eval_id(dataset, eval_key):
    """The evaluation's document id (str), or None."""
    try:
        return str(dataset._doc.evaluations[eval_key].id)
    except (KeyError, AttributeError):
        return None


# Review lifecycle for a run (value -> display label).
STATUSES = {
    "new": "New",
    "candidate": "Candidate",
    "promoted": "Promoted",
    "archived": "Archived",
}


def _get_info(dataset, train_key):
    """The :class:`TrainingInfo` for a key, or ``None`` if absent."""
    if not train_key or not dataset.has_training_run(train_key):
        return None
    return dataset.get_training_info(train_key)


def _num(v):
    """Coerce numpy/float metric values to JSON-safe native numbers."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except Exception:
        return v


def _runs_rows(ctx):
    """All training runs as ``Row`` dicts, newest first, read from the run
    framework via the ``_run_row`` adapter."""
    dataset = ctx.dataset
    rows = [
        _run_row(k, dataset.get_training_info(k))
        for k in dataset.list_training_runs()
    ]
    rows.sort(key=lambda x: (x.get("created_at") or ""), reverse=True)
    return rows


class TrainingRunsPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="training_runs",
            label="Training Runs",
            icon="fitness_center",
            surfaces="grid",
        )

    # --- lifecycle / data push ---------------------------------------------
    def on_load(self, ctx):
        self._push(ctx)

    def _push(self, ctx):
        # set_data writes are synced back to the React component because methods
        # are invoked via the frontend's `useTriggerPanelEvent` (which tags the
        # event with the correct panel_id) -- the same pattern the builtin Model
        # Evaluation panel uses for its reactive data.
        rows = _runs_rows(ctx)
        ctx.panel.set_data("rows", rows)
        ctx.panel.set_data("statuses", STATUSES)
        return rows

    def refresh(self, ctx):
        rows = self._push(ctx)
        n = len(rows)
        ctx.ops.notify(
            f"Refreshed — {n} training run{'' if n == 1 else 's'}",
            variant="success",
        )

    # --- form launchers (auto-refresh via on_success) ----------------------
    def open_log(self, ctx):
        ctx.prompt(f"{_PLUGIN}/log_training_run", on_success=self.refresh)

    def open_edit(self, ctx):
        # Pre-select the run being viewed so the Edit form opens on it directly.
        train_key = ctx.params.get("train_key")
        params = {"train_key": train_key} if train_key else None
        ctx.prompt(
            f"{_PLUGIN}/edit_training_run", params=params, on_success=self.refresh
        )

    # --- navigation actions ------------------------------------------------
    def open_eval(self, ctx):
        # Deep-links the builtin Model Evaluation panel to this eval (does NOT
        # touch the grid view). Three quirks dictate the shape of this call:
        #
        # 1) The payload must go in panel STATE, not data: the React side
        #    renders from a state+data merge either way, but the eval panel's
        #    Python handlers (load_evaluation et al) resolve the key via
        #    ``ctx.panel.get_state("view")``, which only sees state.
        # 2) open_panel writes its ``state`` param WHOLESALE as the panel's
        #    state envelope, of which only the ``state`` key reaches
        #    ``get_state`` -- hence the outer "state" nesting.
        # 3) If the panel is already open, open_panel just focuses it and
        #    drops the payload entirely -- so close it first to force the
        #    initializePanel path.
        #
        # ``init: True`` stops the panel's on_load from resetting the view
        # back to its overview page.
        eval_key = ctx.params.get("eval_key")
        if not eval_key:
            return
        ctx.ops.close_panel(name=_EVAL_PANEL)
        ctx.ops.open_panel(
            _EVAL_PANEL,
            is_active=True,
            layout="horizontal",
            force=True,
            state={
                "state": {
                    "view": {
                        "page": "evaluation",
                        "key": eval_key,
                        "id": _eval_id(ctx.dataset, eval_key),
                        "init": True,
                    }
                }
            },
        )

    def open_view(self, ctx):
        train_key = ctx.params.get("train_key")
        split = ctx.params.get("split") or "train"
        if split not in SPLITS or not ctx.dataset.has_training_run(train_key):
            return
        run = ctx.dataset.load_training_run(train_key)
        view = getattr(run, f"{split}_view", None)
        if view is None:
            return
        ctx.ops.set_view(view)

    # --- mutations ---------------------------------------------------------
    def set_status(self, ctx):
        train_key = ctx.params.get("train_key")
        status = ctx.params.get("status")
        if status not in STATUSES or not ctx.dataset.has_training_run(train_key):
            return
        run = ctx.dataset.load_training_run(train_key)
        run.config.review_status = status
        run.save_config()
        ctx.ops.notify(
            f"Status set to '{STATUSES[status]}'", variant="success"
        )
        self._push(ctx)

    def set_note(self, ctx):
        train_key = ctx.params.get("train_key")
        if not ctx.dataset.has_training_run(train_key):
            return
        run = ctx.dataset.load_training_run(train_key)
        run.config.note = ctx.params.get("note") or None
        run.save_config()
        ctx.ops.notify("Note saved", variant="success")
        self._push(ctx)

    def delete_run(self, ctx):
        train_key = ctx.params.get("train_key")
        if not ctx.dataset.has_training_run(train_key):
            return
        ctx.dataset.delete_training_run(train_key)
        ctx.ops.notify(f"Deleted training run '{train_key}'", variant="success")
        self._push(ctx)

    # --- evaluation summary (metrics rendered in our own Overview) ----------
    def eval_summary(self, ctx):
        """Pushes the eval's high-level metrics (same as ``print_metrics()``:
        accuracy/precision/recall/fscore/support) to the panel. We call only
        ``results.metrics()`` -- NOT ``mAP()``, which recomputes average
        precision over every detection and is slow."""
        train_key = ctx.params.get("train_key")
        eval_key = ctx.params.get("eval_key")
        summary = {"train_key": train_key, "eval_key": eval_key, "metrics": None}

        if eval_key and eval_key in ctx.dataset.list_evaluations():
            try:
                info = ctx.dataset.get_evaluation_info(eval_key)
                results = ctx.dataset.load_evaluation_results(eval_key)
                summary.update(
                    {
                        "type": info.config.type,
                        "method": info.config.method,
                        "gt_field": info.config.gt_field,
                        "pred_field": info.config.pred_field,
                        "metrics": {k: _num(v) for k, v in dict(results.metrics()).items()},
                    }
                )
            except Exception as exc:
                summary["error"] = str(exc)

        ctx.panel.set_data("eval_summary", summary)

    # --- per-split detail (view stages + label distribution) ---------------
    def split_details(self, ctx):
        """Computes, for each recorded split, its view stages and the label-count
        distribution for the run's label field, and pushes it via ``set_data``.

        Membership (count + distribution) comes from the frozen ``*_view_ids``
        (the fact); the displayed stage strings are rehydrated from
        ``*_view_stages`` via the validated ``load_view_stages`` helper (the
        intent)."""
        train_key = ctx.params.get("train_key")
        info = _get_info(ctx.dataset, train_key)
        if info is None:
            ctx.panel.set_data("split_details", {"train_key": train_key, "splits": {}})
            return

        c = info.config
        label_field = c.gt_field
        label_path = None
        if label_field:
            try:
                _, label_path = ctx.dataset._get_label_field_path(label_field, "label")
            except Exception:
                label_path = None

        out = {}
        for split in SPLITS:
            ids = getattr(c, f"{split}_view_ids", None)
            if ids is None:
                continue
            view = ctx.dataset.select(ids)
            distribution = None
            if label_path:
                counts = view.count_values(label_path)
                distribution = sorted(
                    ({"label": str(k), "count": int(v)} for k, v in counts.items()),
                    key=lambda x: -x["count"],
                )
            stages = getattr(c, f"{split}_view_stages", None)
            rebuilt = fout.load_view_stages(ctx.dataset, stages)
            out[split] = {
                "num_samples": view.count(),
                "stages": [str(stage) for stage in rebuilt._all_stages],
                "label_field": label_field,
                "distribution": distribution,
            }

        ctx.panel.set_data(
            "split_details",
            {"train_key": train_key, "splits": out, "label_field": label_field},
        )

    # --- render ------------------------------------------------------------
    def render(self, ctx):
        panel = types.Object()
        return types.Property(
            panel,
            view=types.View(
                component="TrainingRunsView",
                composite_view=True,
                refresh=self.refresh,
                open_log=self.open_log,
                open_edit=self.open_edit,
                open_eval=self.open_eval,
                open_view=self.open_view,
                set_status=self.set_status,
                set_note=self.set_note,
                delete_run=self.delete_run,
                split_details=self.split_details,
                eval_summary=self.eval_summary,
            ),
        )
