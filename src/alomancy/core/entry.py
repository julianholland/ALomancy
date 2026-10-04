"""ALomancy: the single user-facing entry point.

The config decides everything -- the AL skeleton (``general.al_workflow``,
resolved through the registry's "al_workflow" category by
``build_workflow``) and every module it uses (``training.trainer``,
``structure_generation.generator``, ``high_accuracy_evaluation.evaluator``)::

    from alomancy import ALomancy

    ALomancy("config.yaml").run()

A path is read with ``load_dictionaries`` (HPC profile names resolved from
``~/.alomancy/hpc_config.yaml``); a dict is used as-is.
"""

from pathlib import Path
from typing import Any

from alomancy.configs.config_dictionaries import load_dictionaries
from alomancy.core.active_learning_workflow import (
    ActiveLearningWorkflow,
    build_workflow,
)

__all__ = ["ALomancy"]


class ALomancy:
    """Build and run the AL workflow a config describes.

    ``workflow`` is the concrete ``ActiveLearningWorkflow`` subclass the
    config selected; any attribute not defined here (``db``, ``seed``,
    ``num_of_al_loops``, ...) is read from it.
    """

    def __init__(self, config: str | Path | dict[str, Any]):
        if isinstance(config, dict):
            self.jobs_dict = config
        elif isinstance(config, str | Path):
            self.jobs_dict = load_dictionaries(Path(config))
        else:
            raise TypeError(
                "ALomancy(config) takes a path to a YAML config or a config "
                f"dict, got {type(config).__name__}."
            )
        self.workflow: ActiveLearningWorkflow = build_workflow(self.jobs_dict)

    def run(self) -> None:
        """Run the configured workflow."""
        self.workflow.run()

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on ALomancy itself. Guard
        # "workflow" so a half-constructed instance can't recurse.
        if name == "workflow":
            raise AttributeError(name)
        return getattr(self.workflow, name)

    def __repr__(self) -> str:
        jd = self.jobs_dict
        return (
            f"ALomancy(al_workflow={self.workflow.NAME!r}, "
            f"trainer={jd.get('training', {}).get('trainer', 'mace')!r}, "
            f"generator={jd.get('structure_generation', {}).get('generator', 'md')!r}, "
            f"evaluator={jd.get('high_accuracy_evaluation', {}).get('evaluator', 'qe')!r})"
        )
