"""
@voxel51/training-runs

The App-facing window onto training-run lineage. Two ways in:

  - SDK / notebook:  dataset.add_training_run(train_key, train_view, ...)
  - App:             Log Training Run / Edit Training Run forms + Training Runs
                     panel (a React/JS panel; the operators below feed it data
                     and perform its actions)

A run associates views, a checkpoint URI, a tracker URL, and an eval run. This
plugin does NOT run training, and (from the App) it does NOT run eval -- it
links things that already exist. All persistence is the training-run framework
on the dataset (``add_training_run`` / ``load_training_run`` / etc.).
"""

import importlib as _importlib
import sys as _sys

# Reload-safety: FiftyOne re-executes this __init__.py when it reloads plugins,
# but it reuses submodules already cached in sys.modules. Without this, editing
# operators.py/panel.py and then reloading the App raises ImportError for any
# newly added names (the cached submodule is stale). Reload submodules in
# dependency order so edits are picked up without a full server restart.
for _sub in ("operators", "panel"):
    _mod = _sys.modules.get(f"{__name__}.{_sub}")
    if _mod is not None:
        _importlib.reload(_mod)

from .operators import (
    LogTrainingRun,
    EditTrainingRun,
    OpenTrainingRunsPanel,
)
from .panel import TrainingRunsPanel


def register(p):
    p.register(LogTrainingRun)
    p.register(EditTrainingRun)
    p.register(OpenTrainingRunsPanel)
    p.register(TrainingRunsPanel)
