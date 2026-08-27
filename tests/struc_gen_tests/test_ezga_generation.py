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


def test_objective_energy_per_atom():
    objective = objective_energy_per_atom()

    np.testing.assert_allclose(objective(_Dataset()), [-5.0, -6.0])


def test_ezga_config_uses_bounded_mutations_and_per_atom_energy():
    config = build_ezga_config(
        dataset_path=Path("initial.xyz"),
        output_path=Path("output"),
        model_path="model.model",
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


@pytest.mark.parametrize(
    ("min_atoms", "max_atoms"),
    [(0, 41), (10, 9)],
)
def test_ezga_config_rejects_invalid_atom_limits(min_atoms, max_atoms):
    with pytest.raises(ValueError):
        build_ezga_config(
            dataset_path=Path("initial.xyz"),
            output_path=Path("output"),
            model_path="model.model",
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )
