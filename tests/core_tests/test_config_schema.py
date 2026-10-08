"""configs/schema.py: config keys the run never reads are reported as
misplaced (read elsewhere: say where), unused (another module's settings)
or unrecognised, instead of being silently ignored."""

import copy
import logging

import pytest

from alomancy.configs.schema import check_config_keys, format_misplaced
from alomancy.core.active_learning_workflow import (
    _DATASET_KWARGS_KEYS,
    _GENERAL_KNOWN_KEYS,
    build_workflow,
)

_HPC = {"hpc_name": "x", "pre_cmds": [], "partitions": ["p"]}
_BASE = {
    "general": {
        "al_workflow": "committee_uncertainty",
        "elements": ["C"],
        "dataset_kwargs": {"target_config_types": ["init_MP"], "test_ratio": 0.1},
        "committee_uncertainty_kwargs": {"num_of_models_in_committee": 3},
    },
    "initialization": {"hpc": _HPC, "max_time": "1H"},
    "training": {"trainer": "mace", "hpc": _HPC, "max_time": "1H"},
    "structure_generation": {
        "generator": "md",
        "hpc": _HPC,
        "max_time": "1H",
        "structure_selection_kwargs": {"chem_formula_list": None},
        "md_kwargs": {"steps": 100, "structure_selection_kwargs": {"seed": 1}},
    },
    "high_accuracy_evaluation": {"evaluator": "qe", "hpc": _HPC, "max_time": "1H"},
}


def _report(**sections):
    config = copy.deepcopy(_BASE)
    for section, values in sections.items():
        config[section].update(values)
    return check_config_keys(config, _GENERAL_KNOWN_KEYS, set(_DATASET_KWARGS_KEYS))


@pytest.mark.unit
def test_a_complete_config_has_no_findings():
    report = _report()
    assert (report.misplaced, report.unused, report.unrecognised) == ([], [], [])


@pytest.mark.unit
def test_md_seed_settings_in_the_generic_filter_block_point_to_md_kwargs():
    """The real mistake: num_of_md_starts written in the generic seed filter
    (structure_generation.structure_selection_kwargs) was ignored and MD
    ran its default 10 starts."""
    report = _report(
        structure_generation={
            "structure_selection_kwargs": {
                "num_of_md_starts": 20,
                "enforce_chemical_diversity": True,
            }
        }
    )
    assert [(path, key) for path, key, _ in report.misplaced] == [
        (("structure_generation", "structure_selection_kwargs"), "num_of_md_starts"),
        (
            ("structure_generation", "structure_selection_kwargs"),
            "enforce_chemical_diversity",
        ),
    ]
    for _, _, elsewhere in report.misplaced:
        assert elsewhere[0] == (
            "structure_generation",
            "md_kwargs",
            "structure_selection_kwargs",
        )
    assert report.unrecognised == []


@pytest.mark.unit
def test_keys_one_level_too_high_point_to_their_block():
    report = _report(
        structure_generation={"num_of_md_starts": 20, "chem_formula_list": None}
    )
    found = {key: elsewhere[0] for _, key, elsewhere in report.misplaced}
    assert found == {
        "num_of_md_starts": (
            "structure_generation",
            "md_kwargs",
            "structure_selection_kwargs",
        ),
        "chem_formula_list": ("structure_generation", "structure_selection_kwargs"),
    }


@pytest.mark.unit
def test_closest_location_is_suggested_first():
    """seed is read in general and in two other blocks; a seed written in
    the generic seed filter is closest to MD's seed selection."""
    report = _report(structure_generation={"structure_selection_kwargs": {"seed": 5}})
    ((_, key, elsewhere),) = report.misplaced
    assert key == "seed"
    assert elsewhere[0] == (
        "structure_generation",
        "md_kwargs",
        "structure_selection_kwargs",
    )
    assert ("general",) in elsewhere
    message = format_misplaced(
        ("structure_generation", "structure_selection_kwargs"), key, elsewhere
    )
    assert message.startswith(
        "structure_generation.structure_selection_kwargs.seed is not read there"
    )
    assert "belongs in structure_generation.md_kwargs.structure_selection_kwargs" in (
        message
    )


@pytest.mark.unit
def test_typos_are_unrecognised_not_misplaced():
    report = _report(
        general={"num_of_al_loop": 3},
        structure_generation={"md_kwargs": {"stpes": 10}},
    )
    assert report.misplaced == []
    assert report.unrecognised == [
        (("general",), "num_of_al_loop"),
        (("structure_generation", "md_kwargs"), "stpes"),
    ]


@pytest.mark.unit
def test_settings_of_unselected_modules_are_reported_as_unused():
    report = _report(
        training={"trainer": "sevennet", "mace_kwargs": {"max_num_epochs": 5}},
        high_accuracy_evaluation={"vasp_kwargs": {}},
    )
    assert [(path, key) for path, key, _ in report.unused] == [
        (("training",), "mace_kwargs"),
        (("high_accuracy_evaluation",), "vasp_kwargs"),
    ]
    assert "training.trainer is 'sevennet'" in report.unused[0][2]
    assert report.unrecognised == []


@pytest.mark.unit
def test_external_package_settings_are_not_checked():
    """MACE, SevenNet, EZGA, QE and VASP take too many settings to list."""
    report = _report(
        training={"mace_kwargs": {"anything": 1, "r_max": 5.0}},
        structure_generation={
            "generator": "ezga",
            "ezga_kwargs": {"whatever": {"nested": True}},
        },
        high_accuracy_evaluation={"qe_kwargs": {"system": {"ecutwfc": 60}}},
    )
    assert report.misplaced == []
    assert report.unrecognised == []


@pytest.mark.unit
def test_every_run_md_setting_is_known():
    """md_kwargs' keys come from run_md's signature, so all of its
    settings are accepted."""
    report = _report(
        structure_generation={
            "md_kwargs": {
                "steps": 10,
                "temperature": 300,
                "timestep_fs": 0.5,
                "friction": 0.002,
                "ensemble": "npt",
                "pressure": 1.0,
                "equilibration_steps": 5,
                "equilibration_temperature": 300.0,
                "traj_interval": 10,
            }
        }
    )
    assert (report.misplaced, report.unrecognised) == ([], [])


@pytest.mark.unit
def test_workflow_logs_one_warning_per_misplaced_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = copy.deepcopy(_BASE)
    config["structure_generation"]["num_of_md_starts"] = 20
    config["general"]["num_of_al_loop"] = 3

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    module_logger = logging.getLogger("alomancy.core.active_learning_workflow")
    module_logger.addHandler(handler)
    try:
        build_workflow(config)
    finally:
        module_logger.removeHandler(handler)

    by_event = {getattr(r, "event", None): r for r in records}
    misplaced = by_event["config_key_misplaced"]
    assert misplaced.data == {
        "key": "structure_generation.num_of_md_starts",
        "belongs_in": "structure_generation.md_kwargs.structure_selection_kwargs",
    }
    assert "general.num_of_al_loop" in by_event["config_key_unrecognised"].getMessage()
