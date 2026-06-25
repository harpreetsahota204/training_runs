#!/usr/bin/env python3
"""
Validate the `exp-model-train-log` training-run branch end to end.

Exercises the SDK engine against `quickstart` and writes a full report to a .txt
you can paste back for diagnosis:

  * 4.2 write: init -> finish (+auto-eval), linkage, log_predictions, the
    context-manager lifecycle, one-call add_training_run, the door-3 "link
    existing eval" profile, and the 1:1 eval-weld policy.
  * 4.4 read/manage: has/list/get/load/rename/delete.
  * Contracts: eval_key defaults to train_key; no slugging; gt/pred required iff
    an eval runs; auto_eval inference; view-stage rehydration; review/note.
  * App surface (best-effort): the plugin's operators + panel are registered
    (SKIPs if it isn't installed yet -- the engine checks still run).

Each step runs in its own try/except, so one failure is recorded with a full
traceback and the rest continue. Run it with: python validate_training_runs.py
"""

import os
import sys
import platform
import traceback
import datetime

REPORT = "training_runs_validation_report.txt"


# --- tee stdout/stderr into the report file -------------------------------
class _Tee:
    """A stdout/stderr replacement that writes to several streams at once.

    Proxies common stream attributes (encoding, isatty, fileno, ...) to the
    first real stream so libraries that introspect sys.stdout -- e.g.
    FiftyOne/ETA progress bars, which read sys.stdout.encoding -- don't choke
    on the tee.
    """

    def __init__(self, *streams):
        self.streams = streams
        self._primary = streams[0] if streams else None

    def write(self, s):
        # Never let one broken stream (e.g. a closed file) abort the others.
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    def isatty(self):
        # Report non-tty so progress bars render simply and never query a tty.
        return False

    def fileno(self):
        return self._primary.fileno()

    def __getattr__(self, name):
        # Anything we didn't explicitly define -> defer to the primary stream.
        return getattr(self._primary, name)


# --- step harness ---------------------------------------------------------
S = {}            # shared state across steps (dataset, run, key, ...)
RESULTS = []      # (num, name, status) one entry per step, for the summary


class SkipStep(Exception):
    """Raised by a step (usually via ``need``) to mark itself SKIP, not FAIL."""

    pass


def step(num, name, fn):
    """Run one step in isolation, recording PASS / SKIP / FAIL.

    A step is just a function; any exception it raises is caught here so a
    single failure never aborts the rest of the run. The outcome (and a full
    traceback on failure) is printed and appended to ``RESULTS``.
    """
    print("\n" + "=" * 78)
    print("[%s] %s" % (num, name))
    print("-" * 78)
    try:
        fn()
        RESULTS.append((num, name, "PASS"))
        print("✅ [%s] PASS" % num)
    except SkipStep as e:
        RESULTS.append((num, name, "SKIP"))
        print("⏭️  [%s] SKIP: %s" % (num, e))
    except Exception:
        RESULTS.append((num, name, "FAIL"))
        print("❌ [%s] FAIL" % num)
        traceback.print_exc()


def need(*keys):
    """Skip the current step unless every prerequisite is in ``S``.

    Steps stash what they build in ``S`` (e.g. the dataset, the run); a later
    step calls ``need(...)`` so it SKIPs cleanly when an earlier step it
    depends on failed, instead of raising a confusing KeyError.
    """
    for k in keys:
        if k not in S:
            raise SkipStep("prerequisite missing: %s" % k)


# The members we actually care about on each training-run object. probe()
# reports only these (present vs absent) rather than the whole inherited dir().
_RUN_SURFACE = (
    "train_key", "status", "eval_key", "checkpoint_uri", "project_url",
    "auto_eval", "train_config",
    "train_view", "val_view", "test_view", "eval_view", "eval_results",
    "evaluate", "finish", "apply_model", "log_predictions", "link_evaluation",
)
_CONFIG_SURFACE = (
    "train_key", "gt_field", "pred_field", "auto_eval", "status", "eval_key",
    "checkpoint_uri", "project_url", "train_config", "review_status", "note",
    "train_view_ids", "val_view_ids", "test_view_ids",
    "train_view_stages", "val_view_stages", "test_view_stages",
)
_INFO_SURFACE = ("key", "timestamp", "config")


def probe(label, obj, keep):
    """Report only the members we care about (green check / red x).

    Presence is read from ``dir(obj)`` and the callable mark from the class
    attribute, so we never trigger a property getter (e.g. ``eval_view``
    before ``finish``). ``keep`` is the curated surface to check.
    """
    members = set(dir(obj))
    print("  %s (%s):" % (label, type(obj).__name__))
    for name in keep:
        present = name in members
        # () marks a method; properties/attrs print bare.
        mark = "()" if callable(getattr(type(obj), name, None)) else ""
        print("    %-22s %s" % (name + mark, "✅" if present else "⛔️"))


# =========================================================================
# Steps
# =========================================================================
def s0_environment():
    """Confirm we're on the branch: the training API exists on ``fo.Dataset``."""
    import fiftyone as fo

    S["fo"] = fo
    print("python           :", sys.version.split()[0], "on", platform.platform())
    print("fiftyone version :", fo.__version__)
    print("fiftyone path    :", os.path.dirname(fo.__file__))
    print("fo.training ns   :", hasattr(fo, "training"))

    expected = [
        "init_training_run", "list_training_runs", "has_training_run",
        "get_training_info", "load_training_run", "rename_training_run",
        "delete_training_run", "delete_training_runs",
    ]
    present = {m: hasattr(fo.Dataset, m) for m in expected}
    for m, ok in present.items():
        print("  Dataset.%-22s %s" % (m, "✅" if ok else "⛔️"))
    # has_training_runs may be a property -> appears on the class too
    print("  Dataset.%-22s %s" % ("has_training_runs",
          "✅" if hasattr(fo.Dataset, "has_training_runs") else "⛔️"))
    missing = [m for m, ok in present.items() if not ok]
    if missing:
        raise RuntimeError(
            "Missing training API: %s -- you are likely not on "
            "exp-model-train-log (checkout the branch + `pip install -e .`)" % missing
        )


def s1_build_dataset():
    """Clone quickstart into a throwaway dataset with a 70/15/15 tag split."""
    need("fo")
    fo = S["fo"]
    import fiftyone.zoo as foz
    import fiftyone.utils.random as four

    name = "train-runs-validation"
    S["name"] = name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    src = foz.load_zoo_dataset("quickstart")
    dataset = src.clone(name=name)
    dataset.persistent = False
    four.random_split(dataset, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=51)
    S["dataset"] = dataset

    print("fields :", [f for f in dataset.get_field_schema() if not f.startswith("_")])
    for tag in ("train", "val", "test"):
        print("  %-5s %d" % (tag, dataset.match_tags(tag).count()))


def s2_init_run():
    """4.2 write: open a run and confirm the splits + view-stage capture.

    ``init_training_run`` freezes each split to sample IDs and (separately)
    serializes its view stages; this checks both were captured.
    """
    need("fo", "dataset")
    fo = S["fo"]
    dataset = S["dataset"]
    key = "quickstart_baseline"   # MUST be a valid identifier (no hyphens)
    S["key"] = key
    run = dataset.init_training_run(
        train_key=key,
        train_view=dataset.match_tags("train"),
        val_view=dataset.match_tags("val"),
        test_view=dataset.match_tags("test"),
        gt_field="ground_truth",
        pred_field="predictions",
        auto_eval=True,
        project_url="https://wandb.ai/me/quickstart-baseline",
        train_config={"arch": "demo", "lr": 1e-3, "epochs": 0},
    )
    S["run"] = run
    print("status    :", getattr(run, "status", "<no .status>"))
    print("train_key :", getattr(run, "train_key", "<no .train_key>"))
    print("splits    : train=%d val=%d test=%d" % (
        run.train_view.count(), run.val_view.count(), run.test_view.count()))
    print("eval_key  :", getattr(run, "eval_key", "<no .eval_key>"))
    # View stages (intent) are captured alongside ids (fact); the names are
    # None here because live views, not saved-view names, were passed.
    c = run.config
    for split in ("train", "val", "test"):
        stages = getattr(c, "%s_view_stages" % split, "<absent>")
        name = getattr(c, "%s_view_name" % split, "<absent>")
        n = len(stages) if isinstance(stages, list) else stages
        print("  %s_view_stages=%s  %s_view_name=%r" % (split, n, split, name))
        assert isinstance(stages, list) and len(stages) > 0, (
            "%s_view_stages not captured" % split
        )
        assert name is None, "%s_view_name should be None for live views" % split
    # Surface probe -- confirms the run object exposes the members we rely on.
    probe("run", run, _RUN_SURFACE)
    if hasattr(run, "config"):
        probe("run.config", run.config, _CONFIG_SURFACE)


def s3_finish():
    """4.2 write: finish triggers auto-eval; eval_key defaults to train_key."""
    need("run")
    run = S["run"]
    res = run.finish(checkpoint_uri="s3://my-bucket/quickstart-baseline/best.pt")
    S["eval_results"] = res
    print("status        :", getattr(run, "status", "?"))
    print("eval_key      :", getattr(run, "eval_key", "?"))
    print("checkpoint_uri:", getattr(run, "checkpoint_uri", "?"))
    print("eval_view     :", run.eval_view.count(), "samples")
    # Finished + auto-eval ran; eval_key DEFAULTS to train_key (no override).
    assert run.status == "completed", "finish() should leave status=completed"
    assert run.eval_key == S["key"], (
        "eval_key should default to train_key (%r), got %r"
        % (S["key"], run.eval_key)
    )


def s4_linkage():
    """4.2 weld: the auto-eval links back to the run that produced it."""
    need("fo", "dataset", "run", "key")
    dataset = S["dataset"]
    run = S["run"]
    key = S["key"]
    res = S.get("eval_results")
    # Forward: print the eval report (auto-eval here is always a detection eval).
    if res is not None:
        res.print_report()
    # Back-pointer: the eval's config records which run created it.
    ek = run.eval_key
    einfo = dataset.get_evaluation_info(ek)
    back = getattr(einfo.config, "train_key", "<no train_key on eval config>")
    print("eval '%s' -> back-pointer train_key = %s" % (ek, back))
    assert back == key, "back-pointer mismatch: %r != %r" % (back, key)


def s5_discovery():
    """4.4 manage: the has/list discovery surface (incl. the status filter)."""
    need("dataset", "key")
    dataset = S["dataset"]
    key = S["key"]
    # has_training_runs is a property (verified fact); access it directly.
    print("has_training_runs     :", dataset.has_training_runs)
    print("has_training_run(key) :", dataset.has_training_run(key))
    print("list_training_runs()  :", dataset.list_training_runs())
    print("list(status=completed):", dataset.list_training_runs(status="completed"))


def s6_get_info():
    """4.4 manage: config survives the DB round-trip and stages rehydrate.

    Reads back every config field, rebuilds a live view from the stored
    stages and checks it matches the frozen ids, and confirms review_status /
    note round-trip without disturbing the execution status.
    """
    need("dataset", "key")
    dataset = S["dataset"]
    info = dataset.get_training_info(S["key"])
    S["info"] = info
    probe("info", info, _INFO_SURFACE)
    if hasattr(info, "config"):
        probe("info.config", info.config, _CONFIG_SURFACE)
        c = info.config
        for attr in ("eval_key", "checkpoint_uri", "project_url", "train_config",
                     "status", "train_view_ids", "val_view_ids", "test_view_ids",
                     "train_view_stages", "val_view_stages", "test_view_stages",
                     "train_view_name", "val_view_name", "test_view_name"):
            v = getattr(c, attr, "<absent>")
            if attr.endswith("_ids") and isinstance(v, (list, tuple)):
                v = "%d ids" % len(v)
            if attr.endswith("_stages") and isinstance(v, (list, tuple)):
                v = "%d stage(s)" % len(v)
            print("  config.%-18s %s" % (attr, v))
        # Stages survive the DB round-trip (get_training_info reloads) AND are
        # usable, not just stored: rebuild a live DatasetView from the stored
        # stages and check its membership matches the frozen ids.
        import fiftyone.utils.training as fout
        for split in ("train", "val", "test"):
            stages = getattr(c, "%s_view_stages" % split, None)
            assert isinstance(stages, list) and len(stages) > 0, (
                "%s_view_stages lost in DB round-trip" % split
            )
            rebuilt = fout.load_view_stages(dataset, stages)
            ids = getattr(c, "%s_view_ids" % split)
            assert set(rebuilt.values("id")) == set(ids), (
                "%s stages rebuilt a view that does not match the frozen ids"
                % split
            )
        print("  stages rehydrate -> live views matching frozen ids (3/3)")
        # review_status / note default correctly and round-trip when written
        # through the run framework (the path the panel uses).
        assert c.review_status == "new", "review_status should default to 'new'"
        assert c.note is None, "note should default to None"
        run = dataset.load_training_run(S["key"])
        run.config.review_status = "promoted"
        run.config.note = "looks good, promote"
        run.save_config()
        c2 = dataset.get_training_info(S["key"]).config
        assert c2.review_status == "promoted", "review_status did not round-trip"
        assert c2.note == "looks good, promote", "note did not round-trip"
        assert c2.status == "completed", (
            "review_status write must not touch execution status"
        )
        print("  review_status/note round-trip: %r / %r (exec status=%r)" % (
            c2.review_status, c2.note, c2.status))
    print("info.key       :", getattr(info, "key", "<absent>"))
    print("info.timestamp :", getattr(info, "timestamp", "<absent>"))


def s7_reload():
    """4.4 manage: a reloaded run exposes the same status, eval, and views."""
    need("dataset", "key")
    dataset = S["dataset"]
    r = dataset.load_training_run(S["key"])
    print("reloaded type   :", type(r).__name__)
    print("reloaded status :", getattr(r, "status", "?"))
    print("reloaded eval_key:", getattr(r, "eval_key", "?"))
    print("reloaded eval_view:", r.eval_view.count())
    print("reloaded splits  : train=%d val=%d test=%d" % (
        r.train_view.count(), r.val_view.count(), r.test_view.count()))


def s8_log_predictions():
    """4.2 write: the manual log_predictions path + per-sample metric fields.

    Fabricates predictions from ground truth (no model needed) so the
    log_predictions -> finish path and the per-sample metric field it writes
    can be checked offline.
    """
    need("fo", "dataset")
    fo = S["fo"]
    dataset = S["dataset"]
    log_key = "quickstart_logged"
    run2 = dataset.init_training_run(
        train_key=log_key,
        train_view=dataset.match_tags("train"),
        val_view=dataset.match_tags("val"),
        test_view=dataset.match_tags("test"),
        gt_field="ground_truth",
        pred_field="preds_logged",   # NEW field we populate
        auto_eval=True,
    )
    # Fabricate predictions for val+test from ground truth (no model needed).
    eval_split = dataset.match_tags(["val", "test"])
    predictions, metrics = {}, {}
    for s in eval_split.select_fields(["id", "ground_truth"]):
        dets = []
        if s.ground_truth is not None:
            for d in s.ground_truth.detections:
                dets.append(fo.Detection(
                    label=d.label, bounding_box=list(d.bounding_box), confidence=0.9))
        predictions[s.id] = fo.Detections(detections=dets)
        metrics[s.id] = {"img_loss": round(0.1 + 0.01 * len(dets), 3)}
    print("fabricated predictions for %d samples" % len(predictions))
    run2.log_predictions(predictions, metrics=metrics)
    res2 = run2.finish(checkpoint_uri="s3://my-bucket/quickstart-logged/best.pt")
    print("run2 status:", getattr(run2, "status", "?"), "| eval_key:", getattr(run2, "eval_key", "?"))
    mf = "%s_img_loss" % log_key
    print("per-sample metric field '%s' present: %s" % (mf, mf in dataset.get_field_schema()))


def s9_context_manager():
    """4.2 write: the ``with run`` lifecycle -- clean exit finishes, error fails.

    (a) a clean block auto-finishes the run; (b) an exception inside the block
    is re-raised AND recorded (status 'failed' + stored error).
    """
    need("dataset")
    dataset = S["dataset"]
    # (a) clean exit -> auto finish
    with dataset.init_training_run(
        train_key="ctx_clean",
        train_view=dataset.match_tags("train"),
        test_view=dataset.match_tags("test"),
        gt_field="ground_truth", pred_field="predictions",
    ) as _:
        pass
    print("ctx_clean status:", dataset.load_training_run("ctx_clean").status)
    # (b) exception inside -> failed + stored + re-raised
    raised = False
    try:
        with dataset.init_training_run(
            train_key="ctx_boom",
            train_view=dataset.match_tags("train"),
            gt_field="ground_truth", pred_field="predictions",
        ) as _:
            raise RuntimeError("simulated training crash")
    except RuntimeError as e:
        raised = True
        print("caught expected:", e)
    print("exception re-raised:", raised)
    failed = dataset.load_training_run("ctx_boom")
    print("ctx_boom status :", getattr(failed, "status", "?"))
    err = getattr(failed, "error", None) or getattr(getattr(failed, "config", None), "error", None)
    print("error stored    :", bool(err))


def s9b_one_call():
    """4.2 write: add_training_run (init+finish in one call) + saved-view name.

    Passing a saved view BY NAME captures the name breadcrumb;
    association-only (auto_eval=False) needs no gt/pred and runs no eval.
    """
    need("fo", "dataset")
    dataset = S["dataset"]
    if dataset.has_saved_view("val_split"):
        dataset.delete_saved_view("val_split")
    dataset.save_view("val_split", dataset.match_tags("val"))
    run = dataset.add_training_run(
        "quickstart_oneshot",
        train_view=dataset.match_tags("train"),
        val_view="val_split",
        auto_eval=False,
        checkpoint_uri="s3://my-bucket/oneshot/best.pt",
        project_url="https://wandb.ai/me/oneshot",
    )
    c = run.config
    print("status         :", run.status)
    print("eval_key       :", run.eval_key)
    print("checkpoint_uri :", run.checkpoint_uri)
    print("train_view_name:", c.train_view_name)
    print("val_view_name  :", c.val_view_name)
    assert run.status == "completed", "one-call run should finish completed"
    assert run.eval_key is None, "association-only run must not run an eval"
    assert c.train_view_name is None
    assert c.val_view_name == "val_split"
    # The name + stages survive the DB round-trip.
    c2 = dataset.get_training_info("quickstart_oneshot").config
    assert c2.val_view_name == "val_split"
    assert isinstance(c2.val_view_stages, list) and len(c2.val_view_stages) > 0
    print("DB round-trip  : val_view_name=%r, %d val stage(s)" % (
        c2.val_view_name, len(c2.val_view_stages)))


def s10_teardown():
    """4.4 manage: rename + delete the runs (also the test's teardown)."""
    need("dataset")
    dataset = S["dataset"]
    # rename + delete ARE the §4.4 surface under test -- let them bubble.
    dataset.rename_training_run("ctx_clean", "ctx_clean_renamed")
    print("after rename    :", dataset.list_training_runs())
    dataset.delete_training_run("ctx_boom")
    print("after delete one:", dataset.list_training_runs())
    dataset.delete_training_runs()
    print("after delete all:", dataset.list_training_runs())


def s11_door3_link_existing_eval():
    """Door-3 profile: a run that LINKS a pre-existing eval instead of
    running one. No gt/pred, no evaluation executed, back-pointer stamped."""
    need("fo", "dataset")
    dataset = S["dataset"]
    # A standalone eval, created the way a user would by hand (quickstart
    # ships a populated `predictions` field).
    test_view = dataset.match_tags("test")
    test_view.evaluate_detections(
        "predictions", gt_field="ground_truth", eval_key="standalone_eval"
    )
    n_evals_before = len(dataset.list_evaluations())

    run = dataset.add_training_run(
        "door3_log",
        train_view=dataset.match_tags("train"),
        test_view=test_view,
        eval_key="standalone_eval",   # link, don't run
        checkpoint_uri="s3://my-bucket/door3/best.pt",
    )
    S["door3_run"] = run

    print("status        :", run.status)
    print("eval_key      :", run.eval_key)
    print("evals before/after:", n_evals_before, "/", len(dataset.list_evaluations()))
    assert run.status == "completed"
    assert run.eval_key == "standalone_eval"
    assert len(dataset.list_evaluations()) == n_evals_before, (
        "linking must not run a new evaluation"
    )
    # Back-pointer stamped on the EXISTING eval's config.
    back = getattr(
        dataset.get_evaluation_info("standalone_eval").config, "train_key", None
    )
    print("back-pointer  :", back)
    assert back == "door3_log"
    # Forward traversal works.
    assert run.eval_results is not None
    print("eval_view     :", run.eval_view.count(), "samples")
    # auto_eval=True + eval_key is rejected as contradictory.
    try:
        dataset.init_training_run(
            "door3_bad",
            train_view=dataset.match_tags("train"),
            eval_key="standalone_eval",
            auto_eval=True,
            gt_field="ground_truth",
            pred_field="predictions",
        )
        raise AssertionError("auto_eval=True + eval_key should raise")
    except ValueError as e:
        print("contradiction rejected:", e)


def s12_weld_policy():
    """The 1:1 weld: idempotent re-link, refuse on collision, explicit
    overwrite re-stamps and clears the previous owner's forward link."""
    need("dataset", "door3_run")
    dataset = S["dataset"]
    run = S["door3_run"]

    # (a) idempotent re-link of the same pair
    run.link_evaluation("standalone_eval")
    print("idempotent re-link ok; eval_key:", run.eval_key)
    assert run.eval_key == "standalone_eval"

    # (b) collision -> refuse by default, provenance untouched
    rival = dataset.init_training_run(
        "door3_rival", train_view=dataset.match_tags("train")
    )
    refused = False
    try:
        rival.link_evaluation("standalone_eval")
    except ValueError as e:
        refused = True
        print("collision refused:", e)
    assert refused, "linking a claimed eval must refuse by default"
    back = dataset.get_evaluation_info("standalone_eval").config.train_key
    assert back == "door3_log", "refusal must not disturb provenance"

    # (c) nonexistent eval -> refuse
    try:
        rival.link_evaluation("no_such_eval")
        raise AssertionError("nonexistent eval_key should raise")
    except ValueError as e:
        print("nonexistent refused:", e)

    # (d) explicit overwrite -> re-stamps AND clears old owner's forward link
    rival.link_evaluation("standalone_eval", overwrite=True)
    back = dataset.get_evaluation_info("standalone_eval").config.train_key
    old_fwd = dataset.get_training_info("door3_log").config.eval_key
    print("after overwrite: back-pointer=%r, old owner's eval_key=%r" % (back, old_fwd))
    assert back == "door3_rival"
    assert old_fwd is None, "overwrite must clear the previous owner's forward link"
    assert rival.eval_key == "standalone_eval"

    # (e) refused weld at init time must not leave a half-created run behind
    try:
        dataset.init_training_run(
            "door3_thief",
            train_view=dataset.match_tags("train"),
            eval_key="standalone_eval",   # claimed by door3_rival
        )
        raise AssertionError("init with a claimed eval_key should raise")
    except ValueError as e:
        print("init-time collision refused:", e)
    assert not dataset.has_training_run("door3_thief"), (
        "refused weld at init must roll back the run registration"
    )


def s13_key_policy():
    """Input contracts at init time: keys are NOT slugged (a non-identifier
    raises), and gt/pred are required iff THIS engine will run the eval."""
    need("fo", "dataset")
    dataset = S["dataset"]
    # (a) a non-identifier key is rejected -- NO silent slugging (RD1)
    rejected = False
    try:
        dataset.init_training_run(
            train_key="my-bad-key",   # hyphen => not a valid identifier
            train_view=dataset.match_tags("train"),
        )
    except ValueError as e:
        rejected = True
        print("invalid key rejected:", e)
    assert rejected, "a non-identifier train_key must raise (no slugging)"
    assert not dataset.has_training_run("my-bad-key"), (
        "a rejected key must not leave a registered run behind"
    )
    # (b) auto_eval=True requires gt/pred (branch-conditional validation)
    missing = False
    try:
        dataset.init_training_run(
            train_key="needs_fields",
            train_view=dataset.match_tags("train"),
            test_view=dataset.match_tags("test"),
            auto_eval=True,   # engine will run the eval -> gt/pred required
        )
    except ValueError as e:
        missing = True
        print("auto_eval without gt/pred rejected:", e)
    assert missing, "auto_eval=True without gt_field/pred_field must raise"
    assert not dataset.has_training_run("needs_fields")


def s14_auto_eval_inference():
    """auto_eval=None inference: defaults True iff a test_view is present and
    no eval_key is being linked; False otherwise."""
    need("fo", "dataset")
    dataset = S["dataset"]
    # (a) test_view present, no eval_key, auto_eval unspecified -> eval runs
    with dataset.init_training_run(
        train_key="infer_eval",
        train_view=dataset.match_tags("train"),
        test_view=dataset.match_tags("test"),
        gt_field="ground_truth", pred_field="predictions",
    ) as run:
        pass
    print("infer_eval   : auto_eval=%r eval_key=%r" % (
        run.auto_eval, run.eval_key))
    assert run.auto_eval is True, (
        "test_view + no eval_key should infer auto_eval=True"
    )
    assert run.eval_key == "infer_eval", (
        "inferred auto-eval should run and default eval_key to train_key"
    )
    # (b) no test_view -> no eval inferred
    with dataset.init_training_run(
        train_key="infer_noeval",
        train_view=dataset.match_tags("train"),
        gt_field="ground_truth", pred_field="predictions",
    ) as run2:
        pass
    print("infer_noeval : auto_eval=%r eval_key=%r" % (
        run2.auto_eval, run2.eval_key))
    assert run2.auto_eval is False, "no test_view should infer auto_eval=False"
    assert run2.eval_key is None, "no auto-eval should leave eval_key unset"


def s15_plugin_operators():
    """Best-effort: confirm the Training Runs PLUGIN (panel + operators) is
    registered with FiftyOne -- not just the engine on the branch.

    SKIPs (does not fail) when the plugin isn't installed yet, since the
    validator is meant to run right after the branch install. Install the
    plugin per the README, then re-run to validate the App surface too."""
    try:
        import fiftyone.operators as foo
    except Exception as e:
        raise SkipStep("fiftyone.operators unavailable: %s" % e)

    plugin = "@voxel51/training-runs"
    operators = [
        "log_training_run", "edit_training_run", "evaluate_training_run",
        "train_model", "open_training_runs_panel",
    ]
    found = {}
    for op_name in operators + ["training_runs"]:  # last entry is the panel
        uri = "%s/%s" % (plugin, op_name)
        exists = foo.operator_exists(uri, enabled="all")
        found[op_name] = exists
        kind = "panel" if op_name == "training_runs" else "operator"
        print("  %-8s %-44s %s" % (
            kind, uri, "registered" if exists else "MISSING"))

    # Nothing registered -> the plugin isn't installed; SKIP, don't FAIL.
    if not any(found.values()):
        raise SkipStep(
            "no '%s' operators registered -- install the plugin (see "
            "README) and re-run to validate the App surface" % plugin
        )
    # Some registered but not all -> a real problem worth failing on.
    missing = [k for k, ok in found.items() if not ok]
    assert not missing, "plugin registered but missing: %s" % missing


# =========================================================================
def main():
    """Run every step in order, tee output to the report file, print a summary."""
    fh = open(REPORT, "w")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)
    try:
        print("#" * 78)
        print("# 🚂  TRAINING-RUN BRANCH VALIDATION REPORT")
        print("# generated:", datetime.datetime.now().isoformat())
        print("#" * 78)

        step("0", "Environment + API presence", s0_environment)
        step("1", "Build dataset (quickstart + split)", s1_build_dataset)
        step("2", "4.2  init_training_run", s2_init_run)
        step("3", "4.2  finish (+ auto-eval)", s3_finish)
        step("4", "4.2  eval linkage (forward + back-pointer)", s4_linkage)
        step("5", "4.4  discovery (has/list)", s5_discovery)
        step("6", "4.4  get_training_info (+ config probe)", s6_get_info)
        step("7", "4.4  load_training_run (persistence round-trip)", s7_reload)
        step("8", "4.2  log_predictions (manual write path)", s8_log_predictions)
        step("9", "4.2  context-manager lifecycle (clean + failure)", s9_context_manager)
        step("9b", "4.2  add_training_run one-call (+ saved-view name)", s9b_one_call)
        step("10", "4.4  rename + delete (teardown)", s10_teardown)
        step("11", "4.2  door-3: link existing eval (no gt/pred, no exec)", s11_door3_link_existing_eval)
        step("12", "4.2  weld policy (idempotent / refuse / overwrite)", s12_weld_policy)
        step("13", "4.2  key policy + input validation (no slugging)", s13_key_policy)
        step("14", "4.2  auto_eval=None inference (both directions)", s14_auto_eval_inference)
        step("15", "plugin operators + panel registered (App surface)", s15_plugin_operators)

        # Drop the throwaway dataset (best-effort -- don't mask a step failure).
        try:
            if "fo" in S and "name" in S and S["fo"].dataset_exists(S["name"]):
                S["fo"].delete_dataset(S["name"])
                print("\n\U0001f9f9 cleaned up dataset:", S["name"])
        except Exception:
            traceback.print_exc()

        # Summary: one line per step, then the totals and what to do next.
        print("\n" + "#" * 78)
        print("# 📋  SUMMARY")
        print("#" * 78)
        npass = sum(1 for _, _, s in RESULTS if s == "PASS")
        nfail = sum(1 for _, _, s in RESULTS if s == "FAIL")
        nskip = sum(1 for _, _, s in RESULTS if s == "SKIP")
        icon = {"PASS": "✅", "SKIP": "⏭️ ", "FAIL": "❌"}
        for num, name, st in RESULTS:
            print("  %s [%2s] %-6s %s" % (icon[st], num, st, name))
        print("-" * 40)
        print("  PASS=%d  FAIL=%d  SKIP=%d" % (npass, nfail, nskip))
        if nfail:
            print("\n  -> Each FAIL above is a task. Read the traceback and check the")
            print("     call against fiftyone/core/training.py and dataset.py.")
            print("     A FAIL means you are likely NOT on exp-model-train-log.")
        if nskip:
            print("\n  -> SKIP usually means a prerequisite step failed, OR (step 15)")
            print("     the plugin isn't installed yet -- that step is best-effort and")
            print("     does not gate the engine. Install the plugin per the README to")
            print("     light it up.")
        if not nfail:
            print("\n  -> Engine green: the branch delivers the SDK contracts.")
            print("       If step 15 passed too, the App plugin is wired.")
    finally:
        # Always restore the real streams, even if the run blew up.
        sys.stdout, sys.stderr = orig_out, orig_err
        fh.close()
    print("\n📄 Wrote report to: %s" % os.path.abspath(REPORT))
    print("👀 Open the App and inspect the panel.")


if __name__ == "__main__":
    main()
