"""Tests for alomancy.initialize.mp_interface."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_mpr(search_return=None, search_side_effect=None):
    """Build a MagicMock standing in for `with MPRester(key) as mpr: ...`."""
    mock_mpr = MagicMock()
    if search_side_effect is not None:
        mock_mpr.materials.summary.search.side_effect = search_side_effect
    else:
        mock_mpr.materials.summary.search.return_value = (
            search_return if search_return is not None else []
        )
    mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
    mock_mpr.__exit__ = MagicMock(return_value=False)
    return mock_mpr


@pytest.mark.unit
class TestRetrieveMpMaterialDocs:
    def test_raises_without_api_key(self, monkeypatch):
        from alomancy.initialize.mp_interface import retrieve_mp_material_docs

        monkeypatch.delenv("MP_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MP_API_KEY"):
            retrieve_mp_material_docs(["H"], 0.1, 20)

    def test_requests_only_needed_fields(self, monkeypatch):
        """Regression test: requesting all_fields (the mp_api client default)
        pulls in sub-models like bandstructure whose schema can drift out of
        sync with the installed client version, raising a pydantic
        ValidationError on data we never use. Restricting to the fields
        docs_to_atoms actually reads avoids that entirely."""
        from alomancy.initialize.mp_interface import retrieve_mp_material_docs

        monkeypatch.setenv("MP_API_KEY", "fake-key")
        mock_mpr = _mock_mpr()

        with patch("mp_api.client.MPRester", return_value=mock_mpr):
            retrieve_mp_material_docs(["H"], 0.1, 20)

        _, kwargs = mock_mpr.materials.summary.search.call_args
        assert kwargs["fields"] == ["material_id", "structure"]

    def test_queries_all_element_permutations(self, monkeypatch):
        from alomancy.initialize.mp_interface import retrieve_mp_material_docs

        monkeypatch.setenv("MP_API_KEY", "fake-key")
        mock_mpr = _mock_mpr()

        with patch("mp_api.client.MPRester", return_value=mock_mpr):
            retrieve_mp_material_docs(["H", "O"], 0.1, 20)

        # combinations of size 1 (H, O) and size 2 (HO) => 3 calls total
        assert mock_mpr.materials.summary.search.call_count == 3

    def test_docs_collected_across_calls(self, monkeypatch):
        from alomancy.initialize.mp_interface import retrieve_mp_material_docs

        monkeypatch.setenv("MP_API_KEY", "fake-key")
        mock_mpr = _mock_mpr(search_side_effect=[["doc1"], ["doc2"], ["doc3"]])

        with patch("mp_api.client.MPRester", return_value=mock_mpr):
            result = retrieve_mp_material_docs(["H", "O"], 0.1, 20)

        assert result == ["doc1", "doc2", "doc3"]

    def test_search_kwargs_forwarded(self, monkeypatch):
        from alomancy.initialize.mp_interface import retrieve_mp_material_docs

        monkeypatch.setenv("MP_API_KEY", "fake-key")
        mock_mpr = _mock_mpr()

        with patch("mp_api.client.MPRester", return_value=mock_mpr):
            retrieve_mp_material_docs(["H"], 0.25, 30)

        _, kwargs = mock_mpr.materials.summary.search.call_args
        assert kwargs["elements"] == ("H",)
        assert kwargs["num_elements"] == 1
        assert kwargs["energy_above_hull"] == (0, 0.25)
        assert kwargs["num_sites"] == (2, 30)


@pytest.mark.unit
class TestDocsToAtoms:
    def test_converts_structure_field(self):
        from alomancy.initialize.mp_interface import docs_to_atoms

        mock_structure = MagicMock()
        mock_doc = MagicMock(structure=mock_structure)

        mock_atoms = MagicMock()
        with patch(
            "pymatgen.io.ase.AseAtomsAdaptor.get_atoms", return_value=mock_atoms
        ) as mock_get_atoms:
            result = docs_to_atoms([mock_doc])

        mock_get_atoms.assert_called_once_with(mock_structure, msonable=False)
        assert result == [mock_atoms]


@pytest.mark.unit
class TestAtomsListFromMp:
    def test_sets_config_type_and_relaxation_flag(self, monkeypatch):
        from ase import Atoms

        from alomancy.initialize.mp_interface import atoms_list_from_mp

        monkeypatch.setenv("MP_API_KEY", "fake-key")
        atoms = Atoms("H", positions=[[0, 0, 0]])

        with (
            patch(
                "alomancy.initialize.mp_interface.retrieve_mp_material_docs",
                return_value=["doc"],
            ),
            patch(
                "alomancy.initialize.mp_interface.docs_to_atoms",
                return_value=[atoms],
            ),
        ):
            result = atoms_list_from_mp(["H"], 0.1, 20, relax_structures=True)

        assert result[0].info["config_type"] == "init_MP"
        assert result[0].info["needs_relaxation"] is True

    def test_needs_relaxation_defaults_false(self, monkeypatch):
        from ase import Atoms

        from alomancy.initialize.mp_interface import atoms_list_from_mp

        monkeypatch.setenv("MP_API_KEY", "fake-key")
        atoms = Atoms("H", positions=[[0, 0, 0]])

        with (
            patch(
                "alomancy.initialize.mp_interface.retrieve_mp_material_docs",
                return_value=["doc"],
            ),
            patch(
                "alomancy.initialize.mp_interface.docs_to_atoms",
                return_value=[atoms],
            ),
        ):
            result = atoms_list_from_mp(["H"], 0.1, 20)

        assert result[0].info["needs_relaxation"] is False
