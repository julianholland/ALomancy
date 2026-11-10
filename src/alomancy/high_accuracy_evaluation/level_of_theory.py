"""Level of theory: a code-neutral record of how a reference label was made.

Every high-accuracy evaluator (registry category ``dft_evaluator``) has a
``level_of_theory(config, hpc)`` entry point that fills a LevelOfTheory from
its *effective* settings. The orchestrator stamps it on every result
(``atoms.info["REF_level_of_theory"]``, canonical JSON, and
``atoms.info["REF_level_of_theory_id"]``, a short hash of the identity
fields), and the workflow warns when the database mixes labels from
different levels of theory (``level_of_theory_mismatch``).

The fields are chosen to fit any electronic-structure code ASE drives, not
only the ones ALomancy implements. Which keyword of each code fills which
field (QE, VASP, ORCA implemented; FHI-aims, CASTEP, CP2K, Gaussian, DFTB+
written down for when they are added) is the table in
docs/architecture/high_accuracy_evaluation.md, section "Level of theory and
adding an evaluator". A field the backend can't determine is None
("unspecified"), never a guess.

Identity fields decide which labels can share a training set: two
structures whose identity fields differ have energies on different
scales. Precision fields (k-point spacing, smearing, SCF threshold,
numerical grids) are recorded but don't change the id.
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any

logger = logging.getLogger(__name__)

# atoms.info keys the orchestrator stamps on every result.
LEVEL_OF_THEORY_KEY = "REF_level_of_theory"
LEVEL_OF_THEORY_ID_KEY = "REF_level_of_theory_id"

# 1 Rydberg in eV (QE cutoffs, smearing and thresholds are in Ry).
RYDBERG_EV = 13.605693122994


@dataclass
class LevelOfTheory:
    """See the module docstring; per-code sources in
    docs/architecture/high_accuracy_evaluation.md."""

    # Program: qe, vasp, orca (future: aims, castep, cp2k, gaussian, dftbplus).
    code: str
    # semi_empirical | mean_field_hf | dft | correlated | multireference.
    method_class: str | None = None
    # Canonical functional or wavefunction method (PBE, r2SCAN, HSE06,
    # B3LYP, RI-MP2, DLPNO-CCSD(T), GFN2-xTB, DFTB3).
    method: str | None = None
    # DFT only: LDA | GGA | meta-GGA | hybrid | range-separated hybrid |
    # double-hybrid (functional_rung).
    rung: str | None = None
    # none | D2 | D3(0) | D3(BJ) | D4 | TS | MBD | XDM | vdW-DF... | built-in.
    dispersion: str | None = None
    # {element or index: U settings}; {} means no +U.
    hubbard_u: dict | None = None
    # plane_wave | gaussian | numeric_atom_centred | gaussian_plane_wave |
    # slater_koster.
    basis_type: str | None = None
    # Basis name or tier (def2-TZVP, DZVP-MOLOPT-SR-GTH, aims "tight",
    # DFTB+ "3ob-3-1"); None for plane waves.
    basis: str | None = None
    # Orbital (plane-wave) cutoff, eV.
    cutoff_eV: float | None = None
    # Density / grid cutoff, eV (QE ecutrho, CP2K cutoff).
    density_cutoff_eV: float | None = None
    # paw | ultrasoft | norm_conserving | mixed | all_electron | ecp |
    # in_parametrisation.
    core_treatment: str | None = None
    # Per-element pseudopotential / POTCAR / ECP identifiers.
    pseudopotentials: dict | None = None
    # none | in_pseudopotential | spin_orbit | scalar_ZORA | atomic_ZORA |
    # DKH2 | X2C.
    relativity: str | None = None
    # restricted | collinear | noncollinear | per_structure (closed-shell
    # restricted, open-shell unrestricted, by multiplicity).
    spin_treatment: str | None = None
    # none | CPCM(water) | SMD(water) | VASPsol | SCCS | MPE.
    solvation: str | None = None
    # Energy-shifting approximations and unrecognised keywords (RIJCOSX,
    # def2/J, TightPNO), sorted.
    approximations: list[str] = field(default_factory=list)
    # -- precision fields (recorded, not part of the id) --
    # k-point spacing as generate_kpts takes it (1/Angstrom, 2*pi included).
    k_spacing: float | None = None
    # {"scheme": ..., "width_eV": ...}; None when not used.
    smearing: dict | None = None
    # SCF convergence threshold (eV), or the code's keyword (TightSCF).
    scf_threshold: float | str | None = None
    # Other numerical settings (VASP PREC/LREAL, ORCA grids).
    numerical_settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys), as stored in atoms.info."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "LevelOfTheory":
        data = json.loads(text)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def identity(self) -> dict[str, Any]:
        data = self.to_dict()
        return {k: data[k] for k in IDENTITY_FIELDS}

    def identity_id(self) -> str:
        """``lot-`` and 12 hex characters of the sha256 of the identity
        fields. The prefix keeps it a string through extxyz, which reads an
        all-digit value (or digits around one "e") back as a number."""
        text = json.dumps(self.identity(), sort_keys=True, default=str)
        return "lot-" + hashlib.sha256(text.encode()).hexdigest()[:12]

    def label(self) -> str:
        """One line for the run summary, e.g. ``DFT PBE (GGA) +D3(BJ) |
        plane_wave 544 eV | paw | vasp``."""
        method = " ".join(
            part
            for part in (
                (self.method_class or "?").upper().replace("_", "-"),
                self.method or "unspecified method",
                f"({self.rung})" if self.rung else "",
                f"+{self.dispersion}"
                if self.dispersion and self.dispersion != "none"
                else "",
            )
            if part
        )
        basis = " ".join(
            part
            for part in (
                self.basis_type or "",
                self.basis or "",
                f"{self.cutoff_eV:.0f} eV" if self.cutoff_eV is not None else "",
            )
            if part
        )
        parts = [method, basis, self.core_treatment or "", self.code]
        if self.relativity and self.relativity not in ("none", "in_pseudopotential"):
            parts.insert(3, self.relativity)
        if self.solvation and self.solvation != "none":
            parts.insert(3, self.solvation)
        return " | ".join(p for p in parts if p)


PRECISION_FIELDS = ("k_spacing", "smearing", "scf_threshold", "numerical_settings")
IDENTITY_FIELDS = tuple(
    f.name for f in fields(LevelOfTheory) if f.name not in PRECISION_FIELDS
)


def stamp_level_of_theory(atoms_list: list, level: "LevelOfTheory") -> list:
    """Write *level* into each structure's info (LEVEL_OF_THEORY_KEY,
    LEVEL_OF_THEORY_ID_KEY); returns the same list."""
    text, identity = level.to_json(), level.identity_id()
    for atoms in atoms_list:
        atoms.info[LEVEL_OF_THEORY_KEY] = text
        atoms.info[LEVEL_OF_THEORY_ID_KEY] = identity
    return atoms_list


def stored_identity_id(info: dict) -> str | None:
    """The level-of-theory id of a stamped structure, recomputed from its
    stored record (REF_level_of_theory) when there is one, so a change in
    the id's format or an extxyz round trip of the id itself can't make the
    same level look different. Falls back to str() of the stored id; None
    when the structure carries neither."""
    record = info.get(LEVEL_OF_THEORY_KEY)
    if record:
        try:
            return LevelOfTheory.from_json(str(record)).identity_id()
        except (ValueError, TypeError):
            pass
    stored = info.get(LEVEL_OF_THEORY_ID_KEY)
    return None if stored is None else str(stored)


def differing_fields(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Names of the fields whose values differ between two level-of-theory
    dicts (from LevelOfTheory.to_dict or the stored JSON)."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# -- functionals -----------------------------------------------------------

LDA = "LDA"
GGA = "GGA"
META_GGA = "meta-GGA"
HYBRID = "hybrid"
RS_HYBRID = "range-separated hybrid"
DOUBLE_HYBRID = "double-hybrid"

# Normalised name (lower case, no spaces or underscores) -> (canonical
# name, rung). Names the codes accept under several spellings appear once
# per spelling. Extend as needed; an unknown name gets rung None.
_FUNCTIONALS: dict[str, tuple[str, str]] = {}


def _add(rung: str, *names: str) -> None:
    for name in names:
        _FUNCTIONALS[_normalise(name)] = (name, rung)


def _normalise(name: str) -> str:
    return re.sub(r"[\s_]", "", name).lower()


_add(LDA, "LDA", "PZ", "PW92", "VWN", "SVWN", "SPW92")
_add(
    GGA,
    "PBE",
    "PBEsol",
    "revPBE",
    "RPBE",
    "BLYP",
    "BP86",
    "PW91",
    "B97-D",
    "AM05",
    "HCTH",
    "optPBE",
    "optB88",
    "SOGGA11",
    "vdW-DF",
    "vdW-DF2",
    "vdW-DF-cx",
    "rev-vdW-DF2",
    "optB88-vdW",
    "optB86b-vdW",
)
_add(
    META_GGA,
    "SCAN",
    "r2SCAN",
    "rSCAN",
    "TPSS",
    "revTPSS",
    "M06-L",
    "MN15-L",
    "B97M-V",
    "MS2",
    "SCAN+rVV10",
    "r2SCAN+rVV10",
)
_add(
    HYBRID,
    "B3LYP",
    "PBE0",
    "PBEh",
    "B3PW91",
    "TPSSh",
    "M06",
    "M06-2X",
    "MN15",
    "BHandHLYP",
    "X3LYP",
    "SCAN0",
    "r2SCANh",
    "r2SCAN0",
    "B1LYP",
    "PW1PW",
)
_add(
    RS_HYBRID,
    "HSE06",
    "HSE03",
    "HSE",
    "wB97X",
    "wB97X-D",
    "wB97X-D3",
    "wB97X-D4",
    "wB97X-V",
    "wB97M-V",
    "wB97M-D4",
    "CAM-B3LYP",
    "LC-wPBE",
    "LC-BLYP",
)
_add(
    DOUBLE_HYBRID,
    "B2PLYP",
    "B2GP-PLYP",
    "DSD-BLYP",
    "DSD-PBEP86",
    "revDSD-PBEP86",
    "DSD-PBEB95",
    "PWPB95",
    "wB97X-2",
    "wB2PLYP",
)
# Spellings that differ from the canonical name.
_FUNCTIONALS[_normalise("PBE-0")] = ("PBE0", HYBRID)
_FUNCTIONALS[_normalise("HSE-06")] = ("HSE06", RS_HYBRID)
_FUNCTIONALS[_normalise("R2SCAN")] = ("r2SCAN", META_GGA)


def functional_rung(name: str | None) -> tuple[str | None, str | None]:
    """(canonical name, rung) for a functional name in any of the codes'
    spellings; (name as given, None) when it isn't known (logged at
    debug). A trailing dispersion suffix (``-D3BJ``, ``-D3``, ``-D4``,
    ``-D2``) is ignored for the lookup."""
    if not name:
        return None, None
    key = _normalise(name)
    if key in _FUNCTIONALS:
        return _FUNCTIONALS[key]
    stripped = re.sub(r"-(d3bj|d3zero|d3\(bj\)|d3\(0\)|d3|d4|d2)$", "", key)
    if stripped in _FUNCTIONALS:
        return _FUNCTIONALS[stripped]
    logger.debug("Functional %r not in the rung table; rung left unspecified.", name)
    return name, None


def is_hartree_fock(name: str | None) -> bool:
    return bool(name) and _normalise(name or "") in {"hf", "rhf", "uhf", "rohf"}
