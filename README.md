# Training Runs

Track **model-training lineage** in FiftyOne. Every training run becomes one
record on your dataset that answers four questions:

- **What data did the model see?** — the frozen train / val / test views

- **Where is the model?** — a checkpoint URI

- **Where are the experiment notes?** — a tracker URL (W&B, MLflow, …)

- **How good is it?** — a linked FiftyOne evaluation

Those records show up as cards in the **Training Runs** panel, where you can
review them, jump straight to a run's evaluation and failure cases, and kick off new fine-tuning jobs — all without leaving the App.
---

## 1. Install

### a. Install the FiftyOne branch

This plugin needs the `exp-model-train-log` branch of FiftyOne, which adds the training-run engine (`dataset.init_training_run`, `add_training_run`, the eval weld, …). 

Install it from source (Python 3.10–3.12):

```bash
git clone https://github.com/voxel51/fiftyone
cd fiftyone
git checkout exp-model-train-log
bash install.sh          # Windows: .\install.bat
```

See the [branch on GitHub](https://github.com/voxel51/fiftyone/tree/exp-model-train-log).

### b. Install the plugin:

```bash
fiftyone plugins install https://github.com/harpreetsahota204/training_runs
```

To train models from the panel, also install the framework you intend to use
(both are optional — only needed for the "Train Model" door):

```bash
pip install ultralytics        # YOLO (detection / classification /segmentation / pose)

pip install transformers torch # HuggingFace (detection / classification / semantic seg)
```

---

## 2. Confirm everything works

Run the validation harness. It exercises the engine end to end against the
`quickstart` dataset and confirms the plugin's operators are registered:

```bash
cd training_runs
python validate_training_runs.py
```

Expect a green summary (`FAIL=0`). It also writes `training_runs_validation_report.txt` — if anything fails, send that file along.

What it checks (the contracts this plugin relies on):

- **Write path:** `init_training_run → finish` with auto-evaluation, the
  one-call `add_training_run`, `log_predictions`, context-manager lifecycle
  (clean finish + crash → `failed`), the door-3 "link an existing eval" profile, and the 1:1 eval-weld policy (idempotent re-link, refuse on collision, explicit overwrite).

- **Read / manage:** `has` / `list` (incl. `status=` filter) / `get` /
  `load` / `rename` / `delete`.

- **Contracts:** `eval_key` defaults to `train_key`; keys are **not** slugged (a non-identifier raises); `gt`/`pred` are required only when an eval actually runs; `auto_eval=None` inference; view-stage capture and rehydration to live, queryable views; `review_status` / `note` round-trip.

- **App surface (best-effort):** the plugin's five operators + the panel are
  registered. This step **skips** (doesn't fail) if you ran the validator before installing the plugin — the engine checks still run.

---

## 3. Go hands-on

Want to drive the engine yourself and see exactly what gets recorded? Work
through the notebook (`train_model_lineage.ipynb`).

It clones `quickstart`, splits it, fine-tunes a YOLO model for a few epochs, and records the run through the engine — then shows the resulting card, the frozen splits, and the welded evaluation in both directions.

---

## 4. What you can do in the App

Open the panel from the App's panel menu (**+ → Training Runs**), or run the
**List Training Runs** operator.

**The panel** lists every run as a card showing its status, the data it trained on (train/val/test split chips), and its evaluation. From a card you can:

- click into a run to see its frozen splits and load any split as a view;

- **review** a run (new → candidate → promoted → archived) and leave a note;
- open the linked **evaluation** directly in the Model Evaluation panel to walk its failure cases;

- view the experiment in your tracker, evaluate a finished run, or delete it
  (kebab `⋮` menu).

Runs launched from the panel appear immediately and update live through
`SCHEDULED → QUEUED → RUNNING → COMPLETED / FAILED`.

### The operators

| Operator | Label in App | What it does |
|---|---|---|
| `train_model` | **Train Model** | Fine-tune a model and record the run. Task-first: pick a task (detection / classification / segmentation / pose) → a compatible label field → splits → a framework (Ultralytics YOLO or HuggingFace) → model + hyperparameters. Trains, evaluates, and links the result — a finished run is indistinguishable from one created via the SDK. |
| `log_training_run` | **Log Training Run** | Associate a model you trained **elsewhere**: point the run at its views, a checkpoint URI, a tracker URL, and (optionally) an existing evaluation. No training happens. |
| `edit_training_run` | **Edit Training Run** | Update an existing run's checkpoint URI, tracker URL, ground-truth field, or eval link. |
| `evaluate_training_run` | **Evaluate Training Run** | Run a FiftyOne evaluation on a finished run and weld it to the run (type-aware: detection / classification / segmentation / regression). |
| `open_training_runs_panel` | **List Training Runs** | Open the Training Runs panel. |

**Delegated execution:** "Train Model" runs are **delegated by default**, so
they need a worker to pick them up. Start one with:

```bash
 fiftyone delegated launch
```

Without a worker, runs sit in `QUEUED` (shown as a pending card in the panel).

**macOS note:** After training, the trained model is applied to your data to
write predictions back (and run the evaluation). On macOS this prediction
write-back runs **single-process** — FiftyOne's parallel data loading uses the
`spawn` start method, which can't pickle some objects on macOS. This is handled
automatically, so training from the panel works out of the box; you don't need
to set anything. On Linux it uses parallel workers as usual, so the write-back
step is faster there.

On macOS, HuggingFace training additionally falls back to CPU for operations
Apple's MPS backend doesn't support (some detection models, e.g.
deformable-attention DETR), so HF fine-tuning can be slow on a Mac —
Ultralytics YOLO is the smoother macOS path.
