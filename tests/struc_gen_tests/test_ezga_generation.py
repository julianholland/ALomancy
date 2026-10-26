from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from ase import Atoms

from alomancy.structure_generation.ezga.generate_structures import (
    bounded_mutation_add,
    bounded_mutation_remove,
    bounded_mutation_remove_add,
    build_ezga_config,
    filter_new_candidates,
    objective_energy_per_atom,
)
from alomancy.utils.dataset_curation import geometry_digest


class _Dataset:
    def get_all_energies(self):
        return np.array([-10.0, -30.0])

    def get_all_compositions(self, return_species=False):
        compositions = np.array([[2], [5]])
        if return_species:
            return compositions, ["Pd"]
        return compositions


@pytest.mark.unit
def test_objective_energy_per_atom():
    objective = objective_energy_per_atom()

    np.testing.assert_allclose(objective(_Dataset()), [-5.0, -6.0])


@pytest.mark.unit
def test_objective_energy_per_atom_accepts_unevaluated_seeds():
    class DatasetWithNan(_Dataset):
        def get_all_energies(self):
            return np.array([np.nan, -30.0])

    objective = objective_energy_per_atom()

    np.testing.assert_allclose(objective(DatasetWithNan()), [0.0, -6.0])


@pytest.mark.unit
def test_ezga_config_uses_bounded_mutations_and_per_atom_energy():
    config = build_ezga_config(
        dataset_path=Path("initial.xyz"),
        output_path=Path("output"),
        elements=["Pd"],
        calculator_spec_path=Path("output/calculator_spec.json"),
        min_atoms=3,
        max_atoms=40,
        mutation_operators={
            "rattle": {"std": 0.08},
            "random_strain": {"max_strain": 0.03},
        },
    )

    mutations = config["mutation_funcs"]
    mutation_types = [mutation["type"] for mutation in mutations]

    assert config["variation"]["use_magnitude_scaling"] is False
    assert len(mutations) == 5
    assert any(name.endswith("mutation_random_strain") for name in mutation_types)
    assert any(name.endswith("bounded_mutation_add") for name in mutation_types)
    assert any(name.endswith("bounded_mutation_remove") for name in mutation_types)
    assert any(name.endswith("bounded_mutation_remove_add") for name in mutation_types)

    add_config = next(
        mutation
        for mutation in mutations
        if mutation["type"].endswith("bounded_mutation_add")
    )
    remove_config = next(
        mutation
        for mutation in mutations
        if mutation["type"].endswith("bounded_mutation_remove")
    )
    assert add_config["max_atoms"] == 40
    assert remove_config["min_atoms"] == 3
    assert mutations[0]["std"] == 0.08
    assert mutations[1]["max_strain"] == 0.03
    assert config["evaluator"]["objectives_funcs"][0]["type"].endswith(
        "objective_energy_per_atom"
    )
    # The trained model's own calculator, never EZGA's built-in MACE one.
    calculator = config["simulator"]["calculator"]
    assert calculator["type"].endswith("generate_structures.trained_model_calculator")
    assert calculator["calculator_spec_path"] == str(
        Path("output/calculator_spec.json")
    )
    assert calculator["device"] == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("min_atoms", "max_atoms"),
    [(0, 41), (10, 9)],
)
def test_ezga_config_rejects_invalid_atom_limits(min_atoms, max_atoms):
    with pytest.raises(ValueError):
        build_ezga_config(
            dataset_path=Path("initial.xyz"),
            output_path=Path("output"),
            calculator_spec_path=Path("output/calculator_spec.json"),
            elements=["Pd"],
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )


@pytest.mark.unit
def test_ezga_relaxes_with_the_trained_models_own_calculator(tmp_path):
    """The factory named in ezga_config.yaml builds the calculator of the
    trainer that trained the model (on the CPU), wrapped in EZGA's ASE
    adapter -- not EZGA's built-in MACE calculator."""
    import json
    from unittest.mock import MagicMock, patch

    from ase.calculators.emt import EMT

    from alomancy.structure_generation.ezga.generate_structures import (
        trained_model_calculator,
    )

    spec_path = tmp_path / "calculator_spec.json"
    spec_path.write_text(
        json.dumps(
            {"trainer": "sevennet", "trainer_config": {"a": 1}, "model_path": "m.pth"}
        )
    )
    trainer = MagicMock()
    trainer.get_calculator.return_value = EMT()

    with patch("alomancy.mlip.base.get_trainer", return_value=trainer) as get_trainer:
        wrapped = trained_model_calculator(str(spec_path), steps_max=5)

    get_trainer.assert_called_once_with("sevennet", {"a": 1})
    trainer.get_calculator.assert_called_once_with("m.pth", device="cpu")
    assert callable(wrapped)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("operation", "child_count", "accepted"),
    [
        ("add", 4, True),
        ("add", 5, False),
        ("remove", 2, True),
        ("remove", 1, False),
    ],
)
def test_bounded_mutations_handle_ezga_tuple(
    monkeypatch, operation, child_count, accepted
):
    parent = SimpleNamespace(AtomPositionManager=SimpleNamespace(atomCount=3))
    child = SimpleNamespace(AtomPositionManager=SimpleNamespace(atomCount=child_count))

    # Contract returned by EZGA: structure plus proposal metadata.
    trace = {"operation": operation}
    get_log_q_rev = Mock(return_value=-0.7)
    proposal = (child, -0.5, trace, get_log_q_rev)

    mutation = Mock(return_value=proposal)
    monkeypatch.setattr(
        f"ezga.variation.mutation.mutation_{operation}",
        Mock(return_value=mutation),
    )

    if operation == "add":
        apply = bounded_mutation_add(species=["Pd"], max_atoms=4)
    else:
        apply = bounded_mutation_remove(species="Pd", min_atoms=2)

    result = apply(parent)

    mutation.assert_called_once_with(parent)
    if accepted:
        assert result is proposal
    else:
        assert result is None


@pytest.mark.unit
@pytest.mark.parametrize("bound", [[0], [0, "Pd"]])
def test_ezga_add_accepts_atom_indices_in_bound(bound):
    config = build_ezga_config(
        dataset_path=Path("initial.xyz"),
        output_path=Path("output"),
        calculator_spec_path=Path("output/calculator_spec.json"),
        elements=["Pd"],
        mutation_operators={"add": {"bound": bound}},
    )

    mutation = next(
        item
        for item in config["mutation_funcs"]
        if item["type"].endswith("bounded_mutation_add")
    )
    assert mutation["bound"] == bound


@pytest.mark.unit
def test_ezga_remove_add_rejects_bound():
    with pytest.raises(
        ValueError,
        match=r"remove_add\.bound is not supported",
    ):
        build_ezga_config(
            dataset_path=Path("initial.xyz"),
            output_path=Path("output"),
            calculator_spec_path=Path("output/calculator_spec.json"),
            elements=["Pd"],
            mutation_operators={"remove_add": {"bound": [0]}},
        )


@pytest.mark.unit
def test_filter_new_candidates_keeps_only_valid_new_structures():
    seed = Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 1.0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    valid_1 = Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 2.0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    valid_2 = Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 3.0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    wrong_count = Atoms(
        "H",
        positions=[[0, 0, 0]],
        cell=[10, 10, 10],
        pbc=True,
    )
    wrong_element = Atoms(
        "He2",
        positions=[[0, 0, 0], [0, 0, 2.0]],
        cell=[10, 10, 10],
        pbc=True,
    )

    candidates, summary = filter_new_candidates(
        [
            seed.copy(),
            valid_1,
            valid_1.copy(),
            wrong_count,
            wrong_element,
            valid_2,
        ],
        {geometry_digest(seed)},
        elements=["H"],
        min_atoms=2,
        max_atoms=2,
    )

    assert [geometry_digest(atoms) for atoms in candidates] == [
        geometry_digest(valid_1),
        geometry_digest(valid_2),
    ]
    assert all(atoms.info["structure_generator"] == "ezga" for atoms in candidates)
    assert summary == {
        "exported": 6,
        "excluded_seeds": 1,
        "excluded_batch_duplicates": 1,
        "new_candidates": 2,
        "excluded_atom_count": 1,
        "excluded_elements": 1,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("child_count", "accepted"),
    [
        (2, True),
        (4, True),
        (1, False),
        (5, False),
    ],
)
def test_bounded_remove_add_preserves_tuple_and_atom_limits(
    monkeypatch, child_count, accepted
):
    parent = SimpleNamespace(AtomPositionManager=SimpleNamespace(atomCount=3))
    child = SimpleNamespace(AtomPositionManager=SimpleNamespace(atomCount=child_count))

    trace = {"operation": "remove_add"}
    get_log_q_rev = Mock(return_value=-0.7)
    proposal = (child, -0.5, trace, get_log_q_rev)

    mutation = Mock(return_value=proposal)
    mutation_factory = Mock(return_value=mutation)
    monkeypatch.setattr(
        "ezga.variation.mutation.mutation_remove_add",
        mutation_factory,
    )

    apply = bounded_mutation_remove_add(
        species_add=["O"],
        species_remove=["Ni"],
        min_atoms=2,
        max_atoms=4,
        collision_tolerance=1.5,
        slab=True,
    )
    result = apply(parent)

    mutation_factory.assert_called_once_with(
        species_add=["O"],
        species_remove=["Ni"],
        collision_tolerance=1.5,
        slab=True,
    )
    mutation.assert_called_once_with(parent)

    if accepted:
        assert result is proposal
    else:
        assert result is None
