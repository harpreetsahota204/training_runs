"""
@voxel51/training-runs

A plugin POC for training-run lineage. Two ways in:

  - SDK / notebook:  add_training_run(view, train_key, ...)
  - App:             Log Training Run / Edit Training Run forms + Training Runs
                     panel (a React/JS panel; the operators below feed it data
                     and perform its actions)

The record associates a view, a checkpoint URI, a tracker URL, and an eval run.
It does NOT run training, and (from the App) it does NOT run eval -- it links
things that already exist. See lineage.py for the data model.
"""

import importlib as _importlib
import sys as _sys

# Reload-safety: FiftyOne re-executes this __init__.py when it reloads plugins,
# but it reuses submodules already cached in sys.modules. Without this, editing
# lineage.py/operators.py and then reloading the App raises ImportError for any
# newly added names (the cached submodule is stale). Reload submodules in
# dependency order (operators imports from lineage) so edits are picked up
# without a full server restart.
for _sub in ("lineage", "operators", "panel"):
    _mod = _sys.modules.get(f"{__name__}.{_sub}")
    if _mod is not None:
        _importlib.reload(_mod)

from .operators import (
    LogTrainingRun,
    EditTrainingRun,
    OpenTrainingRunsPanel,
)
from .panel import TrainingRunsPanel

# Public SDK surface (importable in a notebook)
from .lineage import (  # noqa: F401
    add_training_run,
    list_training_runs,
    get_training_run,
    get_lineage_for_eval,
    update_training_run,
    delete_training_run,
)


def register(p):
    p.register(LogTrainingRun)
    p.register(EditTrainingRun)
    p.register(OpenTrainingRunsPanel)
    p.register(TrainingRunsPanel)
