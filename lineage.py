"""
Core lineage logic: the data model, store I/O, view capture, and the SDK surface.

The record (one per train_key), stored in a dataset-scoped execution store under
key ``run:<train_key>``, plus a reverse index ``evalidx:<eval_key> -> train_key``:

    {
        "train_key":         str,
        # the train/val/test splits the model used. Each split is captured both
        # ways: serialized stages (intent) + frozen sample ids (fact). val/test
        # are optional. (train's saved-view key is ``saved_view_name`` for
        # backward compatibility; val/test use ``<split>_saved_view_name``.)
        "train_view":        [ ...stage dicts... ],
        "train_sample_ids":  [ ...ids... ],
        "saved_view_name":   str | None,
        "val_view":          [ ...stage dicts... ] | None,
        "val_sample_ids":    [ ...ids... ] | None,
        "val_saved_view_name":  str | None,
        "test_view":         [ ...stage dicts... ] | None,
        "test_sample_ids":   [ ...ids... ] | None,
        "test_saved_view_name": str | None,
        "checkpoint_uri":    str | None,
        "project_url":       str | None,
        "eval_key":          str | None,
        "train_config":      dict | None,
        "created_at":        str,                               # iso8601
    }

Everything is JSON-safe. Association, not execution: the only thing that ever
*runs* is evaluation, and only when the SDK caller passes ``gt_field`` +
``pred_field`` alongside a ``test_view``.
"""

import datetime
import re

import fiftyone as fo
import fiftyone.core.labels as fol

STORE_NAME = "training_runs"
RUN_PREFIX = "run:"
EVALIDX_PREFIX = "evalidx:"

# Per-split record keys: split -> (stages_key, ids_key, saved_view_name_key).
# train's saved-view key is the legacy ``saved_view_name`` for back-compat.
SPLIT_KEYS = {
    "train": ("train_view", "train_sample_ids", "saved_view_name"),
    "val": ("val_view", "val_sample_ids", "val_saved_view_name"),
    "test": ("test_view", "test_sample_ids", "test_saved_view_name"),
}

# Label type on pred_field -> the public evaluate_* method we call (we never
# reimplement evaluation; we call exactly what a user would call by hand).
_EVAL_METHODS = {
    fol.Detections: "evaluate_detections",
    fol.Polylines: "evaluate_detections",
    fol.Keypoints: "evaluate_detections",
    fol.Classification: "evaluate_classifications",
    fol.Classifications: "evaluate_classifications",
    fol.Segmentation: "evaluate_segmentations",
    fol.Regression: "evaluate_regressions",
}


# ---------------------------------------------------------------------------
# Public SDK surface
# ---------------------------------------------------------------------------
def add_training_run(
    samples,
    train_key,
    *,
    val_view=None,
    test_view=None,
    checkpoint_uri=None,
    project_url=None,
    eval_key=None,
    label_field=None,
    gt_field=None,
    pred_field=None,
    train_config=None,
    saved_view_name=None,
    val_saved_view_name=None,
    test_saved_view_name=None,
    eval_kwargs=None,
    store=None,
):
    """Records a training run.

    Captures the train split (``samples``) and, optionally, validation
    (``val_view``) and test (``test_view``) splits -- each both ways: serialized
    stages + frozen sample ids.

    Eval is associated by linking an existing one (``eval_key=``) OR run now by
    passing ``test_view`` together with ``gt_field`` + ``pred_field`` (the eval
    runs on the test split). Passing ``test_view`` alone just logs it.

    Args:
        samples: the training view (a ``Dataset`` or ``DatasetView``)
        train_key: the run identity
        val_view (None): an optional validation split to log
        test_view (None): an optional test split to log (and evaluate on, if
            ``gt_field`` + ``pred_field`` are given)
        store (None): an execution store; derived from the dataset if omitted
            (operators pass ``ctx.store(STORE_NAME)``)

    Returns:
        the record dict
    """
    dataset = samples._dataset
    store = store or _get_store(dataset)

    if eval_key is None and test_view is not None and gt_field and pred_field:
        eval_key = _run_eval(test_view, gt_field, pred_field, train_key, eval_kwargs)

    record = {
        "train_key": train_key,
        "checkpoint_uri": checkpoint_uri,
        "project_url": project_url,
        "eval_key": eval_key,
        "label_field": label_field,
        "train_config": train_config,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _set_split(record, "train", samples, saved_view_name)
    _set_split(record, "val", val_view, val_saved_view_name)
    _set_split(record, "test", test_view, test_saved_view_name)

    _write(store, record)
    return record


def list_training_runs(dataset, store=None):
    """Returns the list of train_keys recorded on the dataset."""
    store = store or _get_store(dataset)
    return [k[len(RUN_PREFIX):] for k in store.list_keys() if k.startswith(RUN_PREFIX)]


def get_training_run(dataset, train_key, store=None):
    """Returns the record dict for a train_key, or None."""
    store = store or _get_store(dataset)
    return store.get(RUN_PREFIX + train_key)


def get_lineage_for_eval(dataset, eval_key, store=None):
    """The primary read path: from an eval_key to its training-run record."""
    store = store or _get_store(dataset)
    train_key = store.get(EVALIDX_PREFIX + eval_key)
    return get_training_run(dataset, train_key, store=store) if train_key else None


def update_training_run(dataset, train_key, store=None, **fields):
    """Merges ``fields`` into an existing record (used by the Edit form).

    ``train_key`` (identity) is not editable and is ignored if passed. Changing
    ``eval_key`` re-points the eval->run index.
    """
    store = store or _get_store(dataset)
    record = get_training_run(dataset, train_key, store=store)
    if record is None:
        raise ValueError(f"no training run '{train_key}'")

    old_eval = record.get("eval_key")
    record.update({k: v for k, v in fields.items() if k != "train_key"})

    if old_eval and old_eval != record.get("eval_key"):
        store.delete(EVALIDX_PREFIX + old_eval)
    _write(store, record)
    return record


def delete_training_run(dataset, train_key, store=None):
    """Removes the record and its eval index entry. Does not touch the eval run."""
    store = store or _get_store(dataset)
    record = get_training_run(dataset, train_key, store=store)
    if record is None:
        return
    if record.get("eval_key"):
        store.delete(EVALIDX_PREFIX + record["eval_key"])
    store.delete(RUN_PREFIX + train_key)


# ---------------------------------------------------------------------------
# View helpers (used by the operators to read records back into views)
# ---------------------------------------------------------------------------
def split_present(record, split):
    """Whether ``record`` has a view logged for the given split."""
    stages_key, _, name_key = SPLIT_KEYS[split]
    return bool(record.get(name_key) or record.get(stages_key))


def load_split_view(dataset, record, split="train"):
    """The view-click target for a split: the saved view if one was chosen,
    else rebuild from the stored stages."""
    stages_key, _, name_key = SPLIT_KEYS[split]
    saved = record.get(name_key)
    if saved:
        return dataset.load_saved_view(saved)
    stages = record.get(stages_key) or []
    if not stages:
        return dataset.view()
    return fo.DatasetView._build(dataset, stages)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _run_eval(test_view, gt_field, pred_field, train_key, eval_kwargs):
    label_type = test_view._get_label_field_type(pred_field)
    method = _EVAL_METHODS.get(label_type)
    if method is None:
        raise ValueError(f"no evaluation method for predictions of type {label_type}")
    # Eval keys must be valid variable names, but train_keys often aren't
    # (e.g. "warehouse-det-2026-05-28"), so sanitize the derived key.
    eval_key = f"{_slugify(train_key)}_eval"
    getattr(test_view, method)(
        pred_field, gt_field=gt_field, eval_key=eval_key, **(eval_kwargs or {})
    )
    return eval_key


def _slugify(train_key):
    slug = re.sub(r"\W+", "_", train_key).strip("_")
    if not slug or slug[0].isdigit():
        slug = "run_" + slug
    return slug


def _set_split(record, split, samples, saved_view_name):
    """Captures ``samples`` into ``record`` under the given split's keys. A
    ``None`` ``samples`` leaves the split empty (val/test are optional)."""
    stages_key, ids_key, name_key = SPLIT_KEYS[split]
    if samples is None:
        record[stages_key] = None
        record[ids_key] = None
        record[name_key] = None
        return
    stages, ids = _capture_view(samples)
    record[stages_key] = stages
    record[ids_key] = ids
    record[name_key] = saved_view_name


def _capture_view(samples):
    view = samples.view() if isinstance(samples, fo.Dataset) else samples
    # ``DatasetView._serialize()`` returns the list of stage dicts directly
    # (JSON-safe). ``DatasetView._build(dataset, stages)`` is the inverse.
    stages = view._serialize()
    ids = view.values("id")
    return stages, ids


def _write(store, record):
    store.set(RUN_PREFIX + record["train_key"], record)
    if record.get("eval_key"):
        store.set(EVALIDX_PREFIX + record["eval_key"], record["train_key"])


def _get_store(dataset):
    """The dataset-scoped execution store. Matches what ``ctx.store(STORE_NAME)``
    returns inside operators/panels, so the SDK and the App share the same
    records."""
    from fiftyone.operators.store import ExecutionStore

    return ExecutionStore.create(STORE_NAME, dataset_id=dataset._doc.id)
