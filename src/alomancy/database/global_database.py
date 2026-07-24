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
        split: str | None = None,
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

        REF_forces (Nx3 array) are serialised into atoms.info before adding
        because sage_lib only persists atoms.info, not atoms.arrays.

        split: optional "train" or "test" tag stored in container metadata.
            Used by get_train_atoms() / get_test_atoms() for restart and
            by remove_redundancy_from_partition for filtered deduplication.

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
                if split is not None:
                    storage_ready_apm.atoms.metadata["split"] = split
                sr_list.append(storage_ready_apm)
                existing.add(key)
                added += 1

        start_id = self.partition.N
        for i, sr in enumerate(sr_list):
            sr.atoms.metadata["global_db_id"] = start_id + i

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

    def get_train_atoms(
        self, exclude_duplicates: bool = True, exclude_high_force: bool = True
    ) -> list[Atoms]:
        """Return all train-split structures, optionally excluding flagged containers."""
        return [
            self._atoms_from_container(c)
            for c in self.partition.list_containers()
            if c.AtomPositionManager.metadata.get("split") == "train"
            and not (
                exclude_duplicates
                and c.AtomPositionManager.metadata.get("is_duplicate", False)
            )
            and not (
                exclude_high_force
                and c.AtomPositionManager.metadata.get("is_high_force", False)
            )
        ]

    def get_test_atoms(self) -> list[Atoms]:
        """Return all test-split structures."""
        return [
            self._atoms_from_container(c)
            for c in self.partition.list_containers()
            if c.AtomPositionManager.metadata.get("split") == "test"
        ]

    def get_split_partition(self, split: str) -> Partition:
        """Export an in-memory Partition containing only structures with the given split tag.

        Used by remove_redundancy_from_partition to work on sage_lib objects
        rather than plain Atoms lists.
        """
        indices = [
            i
            for i, c in enumerate(self.partition.list_containers())
            if c.AtomPositionManager.metadata.get("split") == split
        ]
        return self.partition.export_subset(
            indices, new_path=None, new_storage="memory", batch_size=500, verbose=False
        )

    # ------------------------------------------------------------------
    # In-place metadata update helpers
    # ------------------------------------------------------------------

    def flag_as_duplicates(self, positional_indices: list[int]) -> None:
        """Set is_duplicate=True on containers at the given positional indices.

        Near-duplicates are never deleted from the archive; this flag causes
        get_train_atoms(exclude_duplicates=True) to omit them from XYZ outputs.
        """
        id_meta_map = {i: {"is_duplicate": True} for i in positional_indices}
        self.partition.set_metadata_bulk(id_meta_map, use_indices=True)

    def flag_as_high_force(self, positional_indices: list[int]) -> None:
        """Set is_high_force=True on containers at the given positional indices.

        High-force structures are never deleted from the archive; this flag causes
        get_train_atoms(exclude_high_force=True) to omit them from XYZ outputs.
        """
        id_meta_map = {i: {"is_high_force": True} for i in positional_indices}
        self.partition.set_metadata_bulk(id_meta_map, use_indices=True)

    def assign_global_db_ids(self) -> int:
        """Assign global_db_id to any container that does not already have one.

        Uses the container's 0-based positional index as the ID. Safe to call on
        a DB populated before this feature existed — already-tagged containers are
        skipped. Returns the number of containers newly tagged.
        """
        untagged = {
            i: {"global_db_id": i}
            for i, c in enumerate(self.partition.list_containers())
            if "global_db_id" not in c.AtomPositionManager.metadata
        }
        if untagged:
            self.partition.set_metadata_bulk(untagged, use_indices=True)
        logger.info("assign_global_db_ids: tagged %d container(s).", len(untagged))
        return len(untagged)

    def apply_train_test_split(
        self,
        test_fraction: float = 0.1,
        seed: int = 803,
        eligible_config_types: list[str] | None = None,
    ) -> tuple[int, int]:
        """Apply a random train/test split to all containers that lack a split tag.

        Intended for migrating a DB that was populated before split tagging was
        introduced. Already-tagged containers are left untouched.

        Args:
            test_fraction: Fraction of eligible structures to assign to test.
            seed: Random seed for reproducibility.
            eligible_config_types: If provided, only structures whose config_type
                is in this list are eligible for the test set; all other untagged
                structures are assigned to train unconditionally. If None, every
                untagged structure is eligible.

        Returns:
            (n_train, n_test) — counts of newly tagged containers.
        """
        import random

        eligible_set = set(eligible_config_types) if eligible_config_types is not None else None

        train_only: list[int] = []
        split_eligible: list[int] = []
        for i, c in enumerate(self.partition.list_containers()):
            if c.AtomPositionManager.metadata.get("split"):
                continue  # already tagged
            ct = c.AtomPositionManager.metadata.get("config_type", "")
            if eligible_set is not None and ct not in eligible_set:
                train_only.append(i)
            else:
                split_eligible.append(i)

        if not train_only and not split_eligible:
            logger.info("apply_train_test_split: all containers already tagged, nothing to do.")
            return 0, 0

        meta_map: dict[int, dict] = {i: {"split": "train"} for i in train_only}

        if split_eligible:
            rng = random.Random(seed)
            rng.shuffle(split_eligible)
            n_test = max(1, round(len(split_eligible) * test_fraction))
            test_indices = split_eligible[:n_test]
            train_indices = split_eligible[n_test:]
            meta_map.update({i: {"split": "test"} for i in test_indices})
            meta_map.update({i: {"split": "train"} for i in train_indices})
        else:
            test_indices = []
            train_indices = []

        self.partition.set_metadata_bulk(meta_map, use_indices=True)

        n_train_total = len(train_only) + len(train_indices)
        n_test_total = len(test_indices)
        logger.info(
            "apply_train_test_split: tagged %d train, %d test (%.0f%% test fraction, %d config-type-restricted to train).",
            n_train_total,
            n_test_total,
            100 * test_fraction,
            len(train_only),
        )
        return n_train_total, n_test_total

    def update_splits_post_hoc(
        self, train_atoms: list[Atoms], test_atoms: list[Atoms]
    ) -> int:
        """Match atoms to DB containers by positional hash and set their split tag.

        Called after initialize_training_set to tag structures that were added
        to the DB without a split tag during the DFT evaluation phase. Matching
        uses (config_type, formula, n_atoms, hash-of-rounded-positions) which
        is unique for initialization structures by design.

        Returns the number of containers whose split tag was updated.
        """

        def _key(a: Atoms) -> tuple:
            return (
                a.info.get("config_type", ""),
                a.get_chemical_formula(),
                len(a),
                hash(tuple(a.positions.round(3).flatten().tolist())),
            )

        train_keys = {_key(a) for a in train_atoms}
        test_keys = {_key(a) for a in test_atoms}

        split_meta_map: dict[int, dict] = {}
        for i, container in enumerate(self.partition.list_containers()):
            if container.atoms.metadata.get("split"):
                continue  # already tagged
            a = self._atoms_from_container(container)
            k = _key(a)
            if k in train_keys:
                split_meta_map[i] = {"split": "train"}
            elif k in test_keys:
                split_meta_map[i] = {"split": "test"}

        if split_meta_map:
            self.partition.set_metadata_bulk(split_meta_map, use_indices=True)
        updated = len(split_meta_map)
        logger.debug("update_splits_post_hoc: tagged %d containers.", updated)
        return updated

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
        configure_kwargs: dict = {
            "atomPositions": atoms.positions,
            "atomLabels": atoms.symbols,
            "latticeVectors": atoms.cell,
            "E": energy,
        }
        if forces is not None:
            configure_kwargs["total_force"] = forces
        a.AtomPositionManager.configure(**configure_kwargs)
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
        energy = apm.energy
        forces = apm.forces
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.info.update(meta)
        atoms.info["REF_energy"] = energy
        if forces is not None:
            atoms.arrays["REF_forces"] = forces

        return atoms
