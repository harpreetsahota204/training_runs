"""
@voxel51/training-runs

The App-facing window onto training-run lineage. Three ways in:

  - SDK / notebook:  dataset.add_training_run(train_key, train_view, ...)
  - App, log:        Log / Edit Training Run forms associate views, a checkpoint
                     URI, a tracker URL, and an eval run that already exist.
  - App, train:      The Train Model door fine-tunes an Ultralytics YOLO model
                     and records the result through the engine surface
                     (``init_training_run`` -> ``apply_model`` -> ``finish``).

Everything is fronted by the Training Runs panel (a React/JS panel; the
operators below feed it data and perform its actions). All persistence is the
training-run framework on the dataset (``add_training_run`` /
``load_training_run`` / etc.).
"""

import importlib as _importlib
import sys as _sys

# Reload-safety: FiftyOne re-executes this __init__.py when it reloads plugins,
# but it reuses submodules already cached in sys.modules. Without this, editing
# operators.py/panel.py and then reloading the App raises ImportError for any
# newly added names (the cached submodule is stale). Reload submodules in
# dependency order so edits are picked up without a full server restart.
# `train` reloads after `operators` (it imports helpers from it) and before
# `panel` (which prompts the train operator by URI).
for _sub in ("operators", "train", "panel"):
    _mod = _sys.modules.get(f"{__name__}.{_sub}")
    if _mod is not None:
        _importlib.reload(_mod)

from .operators import (
    LogTrainingRun,
    EditTrainingRun,
    EvaluateTrainingRun,
    OpenTrainingRunsPanel,
)
from .train import TrainModel
from .panel import TrainingRunsPanel


def register(p):
    p.register(LogTrainingRun)
    p.register(EditTrainingRun)
    p.register(EvaluateTrainingRun)
    p.register(TrainModel)
    p.register(OpenTrainingRunsPanel)
    p.register(TrainingRunsPanel)
