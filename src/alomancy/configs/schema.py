"""Which keys each config section actually reads, and a check that sorts
every other key into misplaced, unused or unrecognised.

A key the code never reads is otherwise silently ignored, which hides
mistakes such as ``num_of_md_starts`` written in
``structure_generation.structure_selection_kwargs`` (the generator-agnostic
seed filter) instead of ``structure_generation.md_kwargs.
structure_selection_kwargs`` (MD's own seed selection): the run quietly
falls back to the default.

The schema is a nested dict: ``{key: None}`` for a plain setting, ``{key:
{...}}`` for a block whose keys are checked too, ``{key: OPEN}`` for a block
passed through to another program or validated elsewhere (``hpc``,
filters, ``start_from``, and every external package's settings:
``mace_kwargs``, ``sevennet_kwargs``, ``ezga_kwargs``, ``qe_kwargs``,
``vasp_kwargs`` -- too many keys to list). It is built for the modules the
config selects, from the definitions the code already uses (general key
sets, each workflow's ``KWARGS_DEFAULTS``, and the ``kwargs_schema`` a
module registers for its own settings block; a module without one is
open), so it can't drift from what is read.
"""

from dataclasses import dataclass, field
from typing import Any

from alomancy.registry import registered, resolve

#: A block whose contents aren't checked.
OPEN: Any = object()

_PHASES = (
    "initialization",
    "training",
    "structure_generation",
    "high_accuracy_evaluation",
)
# Read from every phase section (remote_info, the evaluator orchestrator).
_PHASE_COMMON: dict[str, Any] = {"name": None, "max_time": None, "hpc": OPEN}

Path = tuple[str, ...]


@dataclass
class KeyReport:
    """Keys the code does not read, by kind."""

    #: (where it is, key, [where it would be read]), closest location first.
    misplaced: list[tuple[Path, str, list[Path]]] = field(default_factory=list)
    #: (where it is, key, why): settings for a module that isn't selected.
    unused: list[tuple[Path, str, str]] = field(default_factory=list)
    #: (where it is, key): read nowhere.
    unrecognised: list[tuple[Path, str]] = field(default_factory=list)


def _dotted(path: Path, key: str | None = None) -> str:
    return ".".join((*path, key) if key is not None else path)


def config_schema(
    jobs_dict: dict, general_known: set[str], dataset_known: set[str]
) -> tuple[dict[str, Any], dict[Path, dict[str, str]]]:
    """The schema for *jobs_dict*'s selected modules, plus, per section, the
    settings blocks of modules it did not select ({key: reason})."""
    general = jobs_dict.get("general") or {}
    unused: dict[Path, dict[str, str]] = {}

    # general: run settings, the selected workflow's own block.
    workflow_name = general.get("al_workflow", "committee_uncertainty")
    general_schema: dict[str, Any] = dict.fromkeys(general_known)
    general_schema.update(
        dataset_kwargs=dict.fromkeys(dataset_known),
        start_from=OPEN,
        train_filter=OPEN,
        test_filter=OPEN,
        report_suggestions=OPEN,
    )
    unused[("general",)] = {}
    for name in registered("al_workflow"):
        cls = resolve("al_workflow", name).workflow_class
        if not cls.KWARGS_KEY:
            continue
        if name == workflow_name:
            general_schema[cls.KWARGS_KEY] = dict.fromkeys(cls.KWARGS_DEFAULTS)
        else:
            unused[("general",)][cls.KWARGS_KEY] = (
                f"only the {name} workflow reads it; general.al_workflow is "
                f"{workflow_name!r}"
            )

    # initialization: one block per structure type.
    initialiser = resolve("initialiser", "default")
    initialization: dict[str, Any] = {
        **_PHASE_COMMON,
        "read_generated_file": None,
        **{
            namespace: dict.fromkeys(keys)
            for namespace, keys in initialiser.kwargs_schema.items()
        },
    }

    # training: the selected trainer's own block is passed to the backend.
    training_config = jobs_dict.get("training") or {}
    trainer_name = training_config.get("trainer", "mace")
    training: dict[str, Any] = {
        **_PHASE_COMMON,
        "trainer": None,
        "quality_gate": OPEN,
        "device": None,
        "default_dtype": None,
    }
    unused[("training",)] = {}
    for name in registered("mlip_trainer"):
        key = resolve("mlip_trainer", name).trainer_class.KWARGS_KEY
        if name == trainer_name:
            training[key] = OPEN
        else:
            unused[("training",)][key] = (
                f"only the {name} trainer reads it; training.trainer is "
                f"{trainer_name!r}"
            )

    # structure_generation: the selected generator's block is checked.
    sg_config = jobs_dict.get("structure_generation") or {}
    generator_name = sg_config.get("generator", "md")
    structure_generation: dict[str, Any] = {
        **_PHASE_COMMON,
        "generator": None,
        "num_of_structures_to_generate": None,
        "structure_selection_kwargs": dict.fromkeys(
            ("chem_formula_list", "selectable_configs", "atom_number_range")
        ),
    }
    unused[("structure_generation",)] = {}
    for name in registered("structure_generator"):
        key = f"{name}_kwargs"
        if name == generator_name:
            structure_generation[key] = getattr(
                resolve("structure_generator", name), "kwargs_schema", OPEN
            )
        else:
            unused[("structure_generation",)][key] = (
                f"only the {name} generator reads it; "
                f"structure_generation.generator is {generator_name!r}"
            )

    # high_accuracy_evaluation: the selected evaluator's block goes to QE/VASP.
    hae_config = jobs_dict.get("high_accuracy_evaluation") or {}
    evaluator_name = hae_config.get("evaluator", "qe")
    high_accuracy_evaluation: dict[str, Any] = {
        **_PHASE_COMMON,
        "evaluator": None,
        "max_go_time": None,
        "force_ceiling": None,
        "max_num_of_relax_steps": None,
    }
    unused[("high_accuracy_evaluation",)] = {}
    for name in registered("dft_evaluator"):
        key = f"{name}_kwargs"
        if name == evaluator_name:
            high_accuracy_evaluation[key] = OPEN
        else:
            unused[("high_accuracy_evaluation",)][key] = (
                f"only the {name} evaluator reads it; "
                f"high_accuracy_evaluation.evaluator is {evaluator_name!r}"
            )

    schema = {
        "general": general_schema,
        "initialization": initialization,
        "training": training,
        "structure_generation": structure_generation,
        "high_accuracy_evaluation": high_accuracy_evaluation,
        "dataset_curation": OPEN,
    }
    return schema, unused


def _known_paths(schema: dict[str, Any], key: str, path: Path = ()) -> list[Path]:
    """Every block path in *schema* that reads *key*."""
    found = [path] if key in schema else []
    for child_key, child in schema.items():
        if isinstance(child, dict):
            found += _known_paths(child, key, (*path, child_key))
    return found


def _closeness(a: Path, b: Path) -> tuple[int, int]:
    shared = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        shared += 1
    # More shared leading sections first, then the shorter remaining path.
    return (-shared, len(a) + len(b) - 2 * shared)


def check_config_keys(
    jobs_dict: dict, general_known: set[str], dataset_known: set[str]
) -> KeyReport:
    """Every key of *jobs_dict* that its schema doesn't read: misplaced (the
    key is read somewhere else in the schema), unused (settings of a module
    the config doesn't select) or unrecognised (read nowhere)."""
    schema, unused_blocks = config_schema(jobs_dict, general_known, dataset_known)
    report = KeyReport()

    def walk(config: dict, block: dict[str, Any], path: Path) -> None:
        for key, value in config.items():
            if key in block:
                child = block[key]
                if isinstance(child, dict) and isinstance(value, dict):
                    walk(value, child, (*path, key))
                continue
            if key in unused_blocks.get(path, {}):
                report.unused.append((path, key, unused_blocks[path][key]))
                continue
            elsewhere = sorted(
                _known_paths(schema, key), key=lambda p: _closeness(p, path)
            )
            if elsewhere:
                report.misplaced.append((path, key, elsewhere))
            else:
                report.unrecognised.append((path, key))

    walk(jobs_dict, schema, ())
    return report


def format_misplaced(path: Path, key: str, elsewhere: list[Path]) -> str:
    where = _dotted(elsewhere[0]) or "the top level"
    also = (
        f" (also read in {', '.join(_dotted(p) or 'the top level' for p in elsewhere[1:])})"
        if len(elsewhere) > 1
        else ""
    )
    return (
        f"{_dotted(path, key)} is not read there and will be ignored; it "
        f"belongs in {where}{also}."
    )
