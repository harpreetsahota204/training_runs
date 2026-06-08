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

from .lineage import (
    STORE_NAME,
    RUN_PREFIX,
    SPLIT_KEYS,
    delete_training_run,
    get_training_run,
    load_split_view,
    split_present,
    update_training_run,
)

_PLUGIN = "@voxel51/training-runs"
_EVAL_PANEL = "model_evaluation_panel_builtin"


def _eval_id(dataset, eval_key):
    """The evaluation's document id (str), or None."""
    try:
        return str(dataset._doc.evaluations[eval_key].id)
    except Exception:
        return None


# Review lifecycle for a run (value -> display label).
STATUSES = {
    "new": "New",
    "candidate": "Candidate",
    "promoted": "Promoted",
    "archived": "Archived",
}
_DEFAULT_STATUS = "new"


def _num(v):
    """Coerce numpy/float metric values to JSON-safe native numbers."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except Exception:
        return v


def _split_summary(record, split):
    """Compact per-split descriptor for the panel: presence, label, count."""
    stages_key, ids_key, name_key = SPLIT_KEYS[split]
    label = record.get(name_key) or ("ad-hoc view" if record.get(stages_key) else None)
    return {
        "present": split_present(record, split),
        "label": label,
        "num_samples": len(record.get(ids_key) or []),
    }


def _runs_rows(ctx):
    store = ctx.store(STORE_NAME)
    rows = []
    for key in store.list_keys():
        if not key.startswith(RUN_PREFIX):
            continue
        r = store.get(key) or {}
        rows.append(
            {
                "train_key": r.get("train_key"),
                "checkpoint_uri": r.get("checkpoint_uri"),
                "project_url": r.get("project_url"),
                "eval_key": r.get("eval_key"),
                "label_field": r.get("label_field"),
                "created_at": r.get("created_at"),
                "status": r.get("status") or _DEFAULT_STATUS,
                "note": r.get("note") or "",
                "splits": {s: _split_summary(r, s) for s in SPLIT_KEYS},
            }
        )
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
        # Opens the builtin Model Evaluation panel (does NOT touch the grid view).
        # Routes through the ``data`` store only -- the panel's React component
        # reads ``data.view.page`` -- mirroring how the Similarity panel deep-links.
        eval_key = ctx.params.get("eval_key")
        if not eval_key:
            return
        ctx.ops.open_panel(
            _EVAL_PANEL,
            is_active=True,
            layout="horizontal",
            force=True,
            data={
                "view": {
                    "page": "evaluation",
                    "key": eval_key,
                    "id": _eval_id(ctx.dataset, eval_key),
                }
            },
        )

    def open_view(self, ctx):
        train_key = ctx.params.get("train_key")
        split = ctx.params.get("split") or "train"
        record = get_training_run(ctx.dataset, train_key, store=ctx.store(STORE_NAME))
        if not record or split not in SPLIT_KEYS or not split_present(record, split):
            return
        ctx.ops.set_view(view=load_split_view(ctx.dataset, record, split))

    # --- mutations ---------------------------------------------------------
    def set_status(self, ctx):
        train_key = ctx.params.get("train_key")
        status = ctx.params.get("status")
        if not train_key or status not in STATUSES:
            return
        update_training_run(
            ctx.dataset, train_key, store=ctx.store(STORE_NAME), status=status
        )
        ctx.ops.notify(
            f"Status set to '{STATUSES[status]}'", variant="success"
        )
        self._push(ctx)

    def set_note(self, ctx):
        train_key = ctx.params.get("train_key")
        if not train_key:
            return
        update_training_run(
            ctx.dataset,
            train_key,
            store=ctx.store(STORE_NAME),
            note=ctx.params.get("note") or "",
        )
        ctx.ops.notify("Note saved", variant="success")
        self._push(ctx)

    def delete_run(self, ctx):
        train_key = ctx.params.get("train_key")
        if not train_key:
            return
        delete_training_run(ctx.dataset, train_key, store=ctx.store(STORE_NAME))
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
        distribution for the run's label field, and pushes it via ``set_data``."""
        train_key = ctx.params.get("train_key")
        record = get_training_run(ctx.dataset, train_key, store=ctx.store(STORE_NAME))
        if not record:
            ctx.panel.set_data("split_details", {"train_key": train_key, "splits": {}})
            return

        label_field = record.get("label_field")
        label_path = None
        if label_field:
            try:
                _, label_path = ctx.dataset._get_label_field_path(label_field, "label")
            except Exception:
                label_path = None

        out = {}
        for split in SPLIT_KEYS:
            if not split_present(record, split):
                continue
            view = load_split_view(ctx.dataset, record, split)
            distribution = None
            if label_path:
                counts = view.count_values(label_path)
                distribution = sorted(
                    ({"label": str(k), "count": int(v)} for k, v in counts.items()),
                    key=lambda x: -x["count"],
                )
            out[split] = {
                "num_samples": view.count(),
                "stages": [str(stage) for stage in view._all_stages],
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
