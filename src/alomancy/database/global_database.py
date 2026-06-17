import logging
from collections import Counter
from pathlib import Path

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from sage_lib.partition.Partition import Partition
from sage_lib.single_run.SingleRun import SingleRun


_DEFAULT_DEDUP_CONFIG_TYPES = ["IsolatedAtom", "init_MP"]

logger = logging.getLogger(__name__)


class GlobalDatabase:
    """
    Persistent store for all DFT-evaluated structures across the AL workflow.

    Wraps sage_lib's Partition (hybrid storage: HDF5 + SQLite) and adds
    domain-specific helpers for counting by config_type, per-element-combo
    deduplication, and round-tripping ASE Atoms objects (including REF_forces).

    Only post-DFT structures (with REF_energy / REF_forces) should be added.
    The DB is the authoritative source for what has been evaluated; individual
    modules still work with xyz files and are unaware of the DB.
    """

    def __init__(self, db_path: str = "results/global_database") -> None:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.partition = Partition(path=db_path, storage="hybrid")

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_structures(
        self,
        atoms_list: list[Atoms],
        skip_duplicates: bool = True,
        config_types_to_dedup: list[str] | None = None,
    ) -> int:
        """
        Add post-DFT structures to the database.

        Deduplication strategy:
        - For config_types in config_types_to_dedup (default: IsolatedAtom,
          init_MP): exact dedup by (config_type, formula) — prevents adding
          the same element's isolated atom twice if two external datasets
          provide it.
        - All other config_types (dimers, trimers, amorphous, al_loop_N):
          always added; use compute_initialization_needs() for count-based
          checking before generating new structures.

        REF_forces (N×3 array) are serialised into atoms.info before adding
        because sage_lib only persists atoms.info, not atoms.arrays.

        Returns the number of structures actually added.
        """
        if config_types_to_dedup is None:
            config_types_to_dedup = _DEFAULT_DEDUP_CONFIG_TYPES

        existing = self._get_config_type_formula_set() if skip_duplicates else set()
        added = 0
        sr_list = []
        for atoms in atoms_list:
            config_type = atoms.info.get("config_type", "")
            formula = atoms.get_chemical_formula()

            key = (config_type, formula)

            if (
                skip_duplicates
                and config_type in config_types_to_dedup
                and key in existing
            ):
                continue
            storage_ready_apm = self._prepare_for_storage(atoms)

            if storage_ready_apm is not None:
                sr_list.append(storage_ready_apm)
                existing.add(key)
                added += 1

        self.partition.add(sr_list)
        return added

    # ------------------------------------------------------------------
    # Querying / counting
    # ------------------------------------------------------------------

    def count_by_config_type(self) -> dict[str, int]:
        """Return {config_type: count} for all structures in the DB."""
        return {
            ct: sum(counts.values())
            for ct, counts in self.count_all_by_config_type_and_formula().items()
        }

    def count_by_config_type_and_formula(self, config_type: str) -> dict[str, int]:
        """
        Return {formula: count} for all structures with the given config_type.
        Used by compute_initialization_needs for per-element-combo checking.
        """
        counts: Counter[str] = Counter()
        for c in self.partition.list_containers():
            apm = c.AtomPositionManager
            if apm.metadata.get("config_type") == config_type:
                counts[apm.formula] += 1
        return dict(counts)

    def count_all_by_config_type_and_formula(
        self,
    ) -> dict[str, dict[str, int]]:
        """
        Single-pass scan returning {config_type: {formula: count}} for all structures.

        Use this in preference to multiple count_by_config_type_and_formula calls
        when several config_types need to be queried at once.
        """
        result: dict[str, Counter] = {}
        for c in self.partition.list_containers():
            apm = c.AtomPositionManager
            ct = apm.metadata.get("config_type", "")
            if ct not in result:
                result[ct] = Counter()
            result[ct][apm.formula] += 1
        return {ct: dict(counts) for ct, counts in result.items()}

    def get_structures_by_config_type(self, config_types: list[str]) -> list[Atoms]:
        """Return all structures whose config_type is in config_types."""
        return [
            self._atoms_from_container(c)
            for c in self.partition.list_containers()
            if c.AtomPositionManager.metadata.get("config_type") in config_types
        ]

    def get_all_as_atoms(self) -> list[Atoms]:
        """Return all structures in the DB as ASE Atoms objects."""
        return [self._atoms_from_container(c) for c in self.partition.list_containers()]

    @property
    def size(self) -> int:
        return self.partition.N

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config_type_formula_set(self) -> set[tuple[str, str]]:
        """Return set of (config_type, formula) pairs already in the DB."""
        return {
            (ct, formula)
            for ct, formulas in self.count_all_by_config_type_and_formula().items()
            for formula in formulas
        }

    @staticmethod
    def _prepare_for_storage(atoms: Atoms) -> SingleRun | None:
        """
        Return a copy of atoms ready for sage_lib storage, or None to skip.

        Normalises REF_energy / REF_forces from either atoms.info/arrays or a
        calculator.  REF_forces are serialised into info["_REF_forces"] because
        sage_lib only persists atoms.info, not atoms.arrays.  Returns None (and
        logs a warning) if no energy source can be found.
        """
        formula = atoms.get_chemical_formula()
        config_type = atoms.info.get("config_type", "unknown")
        energy = atoms.info.get("REF_energy")
        if energy is None:
            try:
                energy = atoms.get_potential_energy()
            except Exception:
                logger.warning(
                    "No REF_energy and no calculator energy for %s (config_type=%s) — skipping.",
                    formula,
                    config_type,
                )
                return None

        forces = atoms.arrays.get("REF_forces")
        if forces is None:
            try:
                forces = atoms.get_forces()
            except Exception:
                logger.warning(
                    "No REF_forces and no calculator forces for %s (config_type=%s) — storing energy only.",
                    formula,
                    config_type,
                )

        a = SingleRun()
        a.AtomPositionManager.configure(
            atomPositions=atoms.positions,
            atomLabels=atoms.symbols,
            latticeVectors=atoms.cell,
            E=energy,
            total_force=forces,
        )
        a.atoms.metadata = {
            k: v
            for k, v in atoms.info.items()
            if isinstance(v, (str, int, float, bool, list, dict))
        }

        return a

    @staticmethod
    def _atoms_from_container(container) -> Atoms:
        """Reconstruct an ASE Atoms object from a sage_lib SingleRun container."""
        apm = container.AtomPositionManager

        atoms = Atoms(
            symbols=list(apm.atomLabelsList),
            positions=apm.atomPositions,
            cell=apm.latticeVectors,
            pbc=[bool(p) for p in apm.pbc],
        )
        meta = dict(apm.metadata)
        atoms.calc = SinglePointCalculator(atoms, energy=apm.energy, forces=apm.forces)
        atoms.info.update(meta)
        atoms.arrays["REF_forces"] = atoms.get_forces()
        atoms.info["REF_energy"] = atoms.get_potential_energy()

        return atoms
