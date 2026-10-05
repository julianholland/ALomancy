from pathlib import Path

import numpy as np
import pytest

from alomancy.structure_generation.ezga.generate_structures import (
    build_ezga_config,
    objective_energy_per_atom,
)


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
    assert any(name.endswith("mutation_remove_add") for name in mutation_types)

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
