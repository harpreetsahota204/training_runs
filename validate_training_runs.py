#!/usr/bin/env python3
"""
Validate the `exp-model-train-log` training-run branch end to end.

Runs Brian's two SDK halves against the `quickstart` dataset and writes a full
report to a .txt you can paste back for diagnosis:

  * Section 4.2 (write path): init_training_run -> finish (+auto-eval) -> linkage,
    log_predictions, context-manager lifecycle.
  * Section 4.4 (read/manage): has/list/get/load/rename/delete.

Design notes:
  - Every step runs INDEPENDENTLY in its own try/except. One failure does not
    abort the run -- it's recorded with a full traceback and the rest continue,
    so the report is a complete map of what works.
  - All stdout/stderr (including FiftyOne's own prints, e.g. eval reports) is
    tee'd into the report file.
  - It probes the real object surfaces (dir(run), config fields) so naming
    differences from the branch notes are visible.

Usage:
    python validate_training_runs.py
Then send back:  training_runs_validation_report.txt
"""

import os
import sys
import platform
import traceback
import datetime

REPORT = "training_runs_validation_report.txt"

# --- tee stdout/stderr into the report file -------------------------------
class _Tee:
    """Writes to multiple streams. Proxies common stream attributes
    (encoding, isatty, fileno, ...) to the first real stream so libraries
    that introspect sys.stdout -- e.g. FiftyOne/ETA progress bars, which read
    sys.stdout.encoding -- don't choke on the tee."""
    def __init__(self, *streams):
        self.streams = streams
        self._primary = streams[0] if streams else None
    def write(self, s):
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
        # report non-tty so progress bars render simply and never query a tty
        return False
    def fileno(self):
        return self._primary.fileno()
    def __getattr__(self, name):
        # anything we didn't explicitly define -> defer to the primary stream
        return getattr(self._primary, name)

# --- step harness ---------------------------------------------------------
S = {}            # shared state across steps
RESULTS = []      # (num, name, status)

class SkipStep(Exception):
    pass

def step(num, name, fn):
    print("\n" + "=" * 78)
    print("[%s] %s" % (num, name))
    print("-" * 78)
    try:
        fn()
        RESULTS.append((num, name, "PASS"))
        print(">>> [%s] PASS" % num)
    except SkipStep as e:
        RESULTS.append((num, name, "SKIP"))
        print(">>> [%s] SKIP: %s" % (num, e))
    except Exception:
        RESULTS.append((num, name, "FAIL"))
        print(">>> [%s] FAIL" % num)
        traceback.print_exc()

def need(*keys):
    for k in keys:
        if k not in S:
            raise SkipStep("prerequisite missing: %s" % k)

def probe(label, obj):
    """Dump an object's public surface -- helps reconcile API naming."""
    print("  %s : type=%s" % (label, type(obj).__name__))
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    print("  %s.public = %s" % (label, attrs))

# =========================================================================
# Steps
# =========================================================================
def s0_environment():
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
        print("  Dataset.%-22s %s" % (m, "present" if ok else "MISSING"))
    # has_training_runs may be a property -> appears on the class too
    print("  Dataset.%-22s %s" % ("has_training_runs",
          "present" if hasattr(fo.Dataset, "has_training_runs") else "MISSING"))
    missing = [m for m, ok in present.items() if not ok]
    if missing:
        raise RuntimeError(
            "Missing training API: %s -- you are likely not on "
            "exp-model-train-log (checkout the branch + `pip install -e .`)" % missing
        )

def s1_build_dataset():
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
    try:
        print("splits    : train=%d val=%d test=%d" % (
            run.train_view.count(), run.val_view.count(), run.test_view.count()))
    except Exception as e:
        print("split view access error:", e)
    print("eval_key  :", getattr(run, "eval_key", "<no .eval_key>"))
    # view stages (intent) captured alongside ids (fact); names are None
    # because live views, not saved-view names, were passed
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
    # surface probe -- shows the REAL method/attr names on the run object
    probe("run", run)
    if hasattr(run, "config"):
        probe("run.config", run.config)

def s3_finish():
    need("run")
    run = S["run"]
    res = run.finish(checkpoint_uri="s3://my-bucket/quickstart-baseline/best.pt")
    S["eval_results"] = res
    print("status        :", getattr(run, "status", "?"))
    print("eval_key      :", getattr(run, "eval_key", "?"))
    print("checkpoint_uri:", getattr(run, "checkpoint_uri", "?"))
    try:
        print("eval_view     :", run.eval_view.count(), "samples")
    except Exception as e:
        print("eval_view access error:", e)

def s4_linkage():
    need("fo", "dataset", "run", "key")
    dataset = S["dataset"]
    run = S["run"]
    key = S["key"]
    res = S.get("eval_results")
    # forward
    if res is not None:
        try:
            res.print_report()
        except Exception:
            try:
                print("metrics:", res.metrics())
            except Exception as e:
                print("no print_report/metrics:", e)
    # back-pointer
    ek = run.eval_key
    einfo = dataset.get_evaluation_info(ek)
    back = getattr(einfo.config, "train_key", "<no train_key on eval config>")
    print("eval '%s' -> back-pointer train_key = %s" % (ek, back))
    assert back == key, "back-pointer mismatch: %r != %r" % (back, key)

def s5_discovery():
    need("dataset", "key")
    dataset = S["dataset"]
    key = S["key"]
    # has_training_runs: property OR method? report which.
    try:
        val = dataset.has_training_runs
        if callable(val):
            print("has_training_runs : METHOD ->", dataset.has_training_runs())
        else:
            print("has_training_runs : PROPERTY ->", val)
    except Exception as e:
        print("has_training_runs error:", e)
    print("has_training_run(key) :", dataset.has_training_run(key))
    print("list_training_runs()  :", dataset.list_training_runs())
    try:
        print("list(status=completed):", dataset.list_training_runs(status="completed"))
    except Exception as e:
        print("status filter error:", e)

def s6_get_info():
    need("dataset", "key")
    dataset = S["dataset"]
    info = dataset.get_training_info(S["key"])
    S["info"] = info
    probe("info", info)
    if hasattr(info, "config"):
        probe("info.config", info.config)
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
        # stages survive the DB round-trip (get_training_info reloads) AND
        # are usable, not just stored: rebuild a live DatasetView from the
        # stored stages and check it matches the frozen membership
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
        # review_status / note: default correctly, and round-trip when written
        # through the run framework (the path the panel will use)
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
    need("dataset", "key")
    dataset = S["dataset"]
    r = dataset.load_training_run(S["key"])
    print("reloaded type   :", type(r).__name__)
    print("reloaded status :", getattr(r, "status", "?"))
    print("reloaded eval_key:", getattr(r, "eval_key", "?"))
    try:
        print("reloaded eval_view:", r.eval_view.count())
        print("reloaded splits  : train=%d val=%d test=%d" % (
            r.train_view.count(), r.val_view.count(), r.test_view.count()))
    except Exception as e:
        print("reloaded view access error:", e)

def s8_log_predictions():
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
    # fabricate predictions for val+test from ground truth (no model needed)
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
    need("fo", "dataset")
    dataset = S["dataset"]
    # a saved view passed BY NAME -> the name breadcrumb is captured;
    # association-only (auto_eval=False) -> no gt/pred required
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
    # name + stages survive the DB round-trip
    c2 = dataset.get_training_info("quickstart_oneshot").config
    assert c2.val_view_name == "val_split"
    assert isinstance(c2.val_view_stages, list) and len(c2.val_view_stages) > 0
    print("DB round-trip  : val_view_name=%r, %d val stage(s)" % (
        c2.val_view_name, len(c2.val_view_stages)))


def s10_teardown():
    need("dataset")
    dataset = S["dataset"]
    try:
        dataset.rename_training_run("ctx_clean", "ctx_clean_renamed")
        print("after rename    :", dataset.list_training_runs())
    except Exception as e:
        print("rename error:", e)
    try:
        dataset.delete_training_run("ctx_boom")
        print("after delete one:", dataset.list_training_runs())
    except Exception as e:
        print("delete one error:", e)
    dataset.delete_training_runs()
    print("after delete all:", dataset.list_training_runs())

def s11_door3_link_existing_eval():
    """Door-3 profile: a run that LINKS a pre-existing eval instead of
    running one. No gt/pred, no evaluation executed, back-pointer stamped."""
    need("fo", "dataset")
    dataset = S["dataset"]
    # a standalone eval, created the way a user would by hand (quickstart
    # ships a populated `predictions` field)
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
    # back-pointer stamped on the EXISTING eval's config
    back = getattr(
        dataset.get_evaluation_info("standalone_eval").config, "train_key", None
    )
    print("back-pointer  :", back)
    assert back == "door3_log"
    # forward traversal works
    assert run.eval_results is not None
    print("eval_view     :", run.eval_view.count(), "samples")
    # auto_eval=True + eval_key is rejected as contradictory
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


# =========================================================================
def main():
    fh = open(REPORT, "w")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)
    try:
        print("#" * 78)
        print("# TRAINING-RUN BRANCH VALIDATION REPORT")
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

        # final dataset cleanup
        try:
            if "fo" in S and "name" in S and S["fo"].dataset_exists(S["name"]):
                S["fo"].delete_dataset(S["name"])
                print("\ncleaned up dataset:", S["name"])
        except Exception:
            traceback.print_exc()

        # summary
        print("\n" + "#" * 78)
        print("# SUMMARY")
        print("#" * 78)
        npass = sum(1 for _, _, s in RESULTS if s == "PASS")
        nfail = sum(1 for _, _, s in RESULTS if s == "FAIL")
        nskip = sum(1 for _, _, s in RESULTS if s == "SKIP")
        for num, name, st in RESULTS:
            print("  [%2s] %-6s %s" % (num, st, name))
        print("-" * 40)
        print("  PASS=%d  FAIL=%d  SKIP=%d" % (npass, nfail, nskip))
        if nfail or nskip:
            print("\n  -> Failures/skips above are the first task list. For each FAIL,")
            print("     read the traceback and check the call against fiftyone/core/training.py")
            print("     and collections.py. SKIP usually means an earlier step it depended on failed.")
        else:
            print("\n  -> All green. Branch delivers Piece 1 (4.2 + 4.4).")
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        fh.close()
    print("\nWrote report to: %s" % os.path.abspath(REPORT))
    print("Send that file back.")

if __name__ == "__main__":
    main()
