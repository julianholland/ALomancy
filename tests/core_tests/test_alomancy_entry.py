"""Tests for core/entry.py's ALomancy -- the single entry point whose AL
skeleton is chosen by general.al_workflow in the config."""

import subprocess
import sys

import pytest
import yaml

from alomancy.core.committee_uncertainty_workflow import CommitteeUncertaintyWorkflow
from alomancy.core.entry import ALomancy
from alomancy.core.random_selection_workflow import RandomSelectionWorkflow


@pytest.fixture
def config(minimal_jobs_dict):
    minimal_jobs_dict["general"] = {
        "elements": ["H"],
        "plots": False,
        "dataset_kwargs": {"target_config_types": ["IsolatedAtom"], "test_ratio": 0.1},
    }
    minimal_jobs_dict["training"] = minimal_jobs_dict.pop("mlip_committee")
    del minimal_jobs_dict["training"]["num_of_models_in_committee"]
    return minimal_jobs_dict


@pytest.mark.unit
@pytest.mark.parametrize(
    ("al_workflow", "cls"),
    [
        ("committee_uncertainty", CommitteeUncertaintyWorkflow),
        ("random_selection", RandomSelectionWorkflow),
        (None, CommitteeUncertaintyWorkflow),
    ],
)
def test_config_selects_the_skeleton(config, al_workflow, cls):
    if al_workflow is not None:
        config["general"]["al_workflow"] = al_workflow
    assert type(ALomancy(config).workflow) is cls


@pytest.mark.unit
def test_yaml_path_is_loaded_and_dispatched(config, tmp_path):
    config["general"]["al_workflow"] = "random_selection"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    for arg in (path, str(path)):
        al = ALomancy(arg)
        assert type(al.workflow) is RandomSelectionWorkflow
        assert al.jobs_dict["general"]["al_workflow"] == "random_selection"


@pytest.mark.unit
def test_unknown_al_workflow_lists_available(config):
    config["general"]["al_workflow"] = "furthest_point_sampling"
    with pytest.raises(ValueError, match="random_selection"):
        ALomancy(config)


@pytest.mark.unit
def test_rejects_non_config_argument():
    with pytest.raises(TypeError, match="config dict"):
        ALomancy(42)  # type: ignore[arg-type]


@pytest.mark.unit
def test_delegates_attributes_and_run(config, shared_db, monkeypatch):
    al = ALomancy(config)
    al.workflow.db = shared_db
    assert al.db is shared_db
    assert al.seed == al.workflow.seed
    ran = []
    monkeypatch.setattr(al.workflow, "run", lambda: ran.append(True))
    al.run()
    assert ran == [True]
    with pytest.raises(AttributeError):
        al.no_such_attribute  # noqa: B018


@pytest.mark.unit
def test_repr_names_skeleton_and_modules(config):
    config["general"]["al_workflow"] = "random_selection"
    text = repr(ALomancy(config))
    assert "random_selection" in text
    assert "'mace'" in text and "'md'" in text and "'qe'" in text


@pytest.mark.unit
def test_direct_construction_with_mismatched_config_raises(config):
    config["general"]["al_workflow"] = "committee_uncertainty"
    with pytest.raises(ValueError, match="ALomancy"):
        RandomSelectionWorkflow(jobs_dict=config)


@pytest.mark.unit
def test_direct_construction_with_matching_or_absent_key_still_works(config):
    assert isinstance(
        CommitteeUncertaintyWorkflow(jobs_dict=config), CommitteeUncertaintyWorkflow
    )
    config["general"]["al_workflow"] = "committee_uncertainty"
    assert isinstance(
        CommitteeUncertaintyWorkflow(jobs_dict=config), CommitteeUncertaintyWorkflow
    )


@pytest.mark.unit
def test_top_level_export_is_lazy():
    code = (
        "import sys, alomancy; "
        "assert 'alomancy.core' not in sys.modules; "
        "from alomancy import ALomancy; "
        "from alomancy.core.entry import ALomancy as E; "
        "assert ALomancy is E"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_cli_run_builds_and_runs(config, tmp_path, monkeypatch):
    from alomancy.cli.main import main

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    ran = []
    monkeypatch.setattr(ALomancy, "run", lambda self: ran.append(self.workflow.NAME))
    monkeypatch.setattr(sys, "argv", ["alomancy", "run", str(path)])
    main()
    assert ran == ["committee_uncertainty"]
