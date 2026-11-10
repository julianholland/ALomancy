"""ORCA backend (registry ``dft_evaluator`` / ``orca``).

ORCA is a molecular code: no periodic boundary conditions (periodic input
raises), and besides DFT it runs wavefunction methods (MP2, coupled
cluster, multireference) whose cost grows steeply with system size, so
structures above ``orca_kwargs.max_num_of_atoms_correlated`` are dropped
before submission when such a method is configured (max_num_of_atoms).

Parallelism goes in the input file (``%pal nprocs``, ``%maxcore``), not
the launch command: ORCA must be started by its absolute path and spawns
its own ``mpirun`` (OpenMPI comes from the HPC profile's ``pre_cmds``).
Both come from the HPC profile's node_info through
utils/dft_utils.resolve_parallel_resources, unless the user sets them
(``orca_kwargs.nprocs``/``maxcore``, a ``%pal``/``%maxcore`` block in
``orcablocks``, or a ``PALn`` keyword): a user value wins, and raises when
the job can't hold it.

Runs remotely (module-level functions only): changes need
``alomancy upgrade-hpc``.
"""

import logging
import numbers
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.data import ground_state_magnetic_moments

from alomancy.high_accuracy_evaluation.level_of_theory import (
    LevelOfTheory,
    functional_rung,
    is_hartree_fock,
)
from alomancy.utils.dft_utils import (
    _run_go,
    _run_sp,
    resolve_parallel_resources,
)

logger = logging.getLogger(__name__)

# Registry entry points (see registry.py).
MAX_OUTPUT_DIR_LEN: int | None = None
SUPPORTS_PERIODIC = False
SUPPORTS_NON_PERIODIC = True

_WHERE = "high_accuracy_evaluation.orca_kwargs"

ORCA_KWARGS_DEFAULTS: dict[str, Any] = {
    "orcasimpleinput": None,  # required
    "orcablocks": "",
    "charge": 0,
    # None: from the HPC profile (resolve_parallel_resources).
    "nprocs": None,
    "maxcore": None,  # MB per process
    # Correlated / multireference methods only; None turns the cap off.
    "max_num_of_atoms_correlated": 20,
}

# Added to orcasimpleinput when missing: EnGrad makes ORCA write the
# gradient (forces) to orca.engrad, which ASE reads; SCFConvergenceForced
# makes ORCA abort when the SCF doesn't converge instead of carrying on
# and printing energies and forces from an unconverged density.
_REQUIRED_KEYWORDS = ("EnGrad", "SCFConvergenceForced")

# Cost classes (classify_orca_method).
MEAN_FIELD = "mean_field"
SEMI_EMPIRICAL = "semi_empirical"
CORRELATED = "correlated"
MULTIREFERENCE = "multireference"
_CAPPED_COST_CLASSES = frozenset({CORRELATED, MULTIREFERENCE})
# Without analytic gradients in ORCA (forces from finite differences:
# about 6N energy calculations per structure). Checked against the ORCA
# version in use when the run starts (validate_kwargs warns).
_NUMERICAL_GRADIENT_FAMILIES = frozenset(
    {
        "ccsd",
        "ccsd(t)",
        "ccsdt",
        "qcisd",
        "qcisd(t)",
        "cisd",
        "cepa",
        "cpf",
        "acpf",
        "aqcc",
        "bccd",
        "bccd(t)",
        "lccsd",
        "nevpt2",
        "caspt2",
        "mrci",
        "mrci+q",
    }
)

_WAVEFUNCTION_PREFIXES = (
    "ri-",
    "dlpno-",
    "lpno-",
    "scs-",
    "sos-",
    "oo-",
    "local-",
    "sc-",
    "fic-",
    "pc-",
)
_CORRELATED_FAMILIES = frozenset(
    {
        "mp2",
        "mp3",
        "mp4",
        "ccd",
        "ccsd",
        "ccsd(t)",
        "ccsd(t1)",
        "ccsdt",
        "qcisd",
        "qcisd(t)",
        "cisd",
        "cisd(t)",
        "cepa",
        "cepa/1",
        "cpf",
        "cpf/1",
        "acpf",
        "aqcc",
        "lccsd",
        "bccd",
        "bccd(t)",
    }
)
_MULTIREFERENCE_FAMILIES = frozenset(
    {"casscf", "nevpt2", "caspt2", "mrci", "mrci+q", "mrddci", "sorci"}
)
# Blocks that run a multireference / correlated calculation whatever the
# simple input says.
_MULTIREFERENCE_BLOCKS = ("%casscf", "%mrci", "%autoci")
_SEMI_EMPIRICAL = frozenset(
    {
        "xtb",
        "xtb0",
        "xtb1",
        "xtb2",
        "gfn-xtb",
        "gfn0-xtb",
        "gfn1-xtb",
        "gfn2-xtb",
        "native-gfn-xtb",
        "native-gfn1-xtb",
        "native-gfn2-xtb",
        "gfn-ff",
        "am1",
        "pm3",
        "mndo",
        "zindo/s",
    }
)
# Composite "-3c" methods carry their own basis and corrections.
_COMPOSITE_3C = re.compile(r"^(.+)-3c$")

_RUN_KEYWORDS = frozenset(
    {
        "sp",
        "engrad",
        "numgrad",
        "opt",
        "copt",
        "zcopt",
        "gdiis-copt",
        "looseopt",
        "tightopt",
        "verytightopt",
        "freq",
        "numfreq",
        "scfconvergenceforced",
        "miniprint",
        "smallprint",
        "normalprint",
        "largeprint",
        "printbasis",
        "printmos",
        "nopop",
        "keepdens",
        "moread",
        "noautostart",
        "autostart",
        "slowconv",
        "veryslowconv",
        "kdiis",
        "soscf",
        "nososcf",
        "noiter",
        "uno",
        "uco",
    }
)
_SCF_THRESHOLDS = frozenset(
    {
        "sloppyscf",
        "loosescf",
        "normalscf",
        "strongscf",
        "tightscf",
        "verytightscf",
        "extremescf",
    }
)
_RI_APPROXIMATIONS = frozenset(
    {
        "rijcosx",
        "ri-j",
        "rij",
        "rijk",
        "ri-jk",
        "rijonx",
        "nori",
        "nocosx",
        "cosx",
        "ri",
        "autoaux",
        "loosepno",
        "normalpno",
        "tightpno",
        "frozencore",
        "nofrozencore",
    }
)
_RELATIVITY = {"zora": "scalar_ZORA", "dkh": "DKH2", "dkh2": "DKH2", "x2c": "X2C"}
_DISPERSION = {
    # ORCA reads plain D3 as D3BJ.
    "d3": "D3(BJ)",
    "d3bj": "D3(BJ)",
    "d3zero": "D3(0)",
    "d4": "D4",
    "d2": "D2",
    "nl": "VV10",
    "scnl": "VV10(SC)",
}
_SPIN = {
    "rks": "restricted",
    "rhf": "restricted",
    "uks": "unrestricted",
    "uhf": "unrestricted",
    "roks": "restricted_open",
    "rohf": "restricted_open",
}
_SOLVATION = re.compile(r"^(cpcm|cpcmc|smd|alpb|ddcosmo)(\(.*\))?$")
_GRID = re.compile(r"^(defgrid\d|grid\d+|finalgrid\d+|nofinalgrid|gridx\d+)$")
_BASIS_PREFIXES = (
    "def2-",
    "def-",
    "ma-def2",
    "zora-def2",
    "dkh-def2",
    "old-",
    "cc-p",
    "aug-cc",
    "jun-cc",
    "may-cc",
    "apr-cc",
    "6-31",
    "6-311",
    "3-21",
    "sto-",
    "pcseg",
    "aug-pcseg",
    "pc-",
    "x2c-",
    "sarc",
    "minix",
    "ano-",
    "saug-",
    "pcsseg",
)
_BASIS_NAMES = frozenset(
    {"svp", "tzvp", "tzvpp", "qzvp", "qzvpp", "minis", "midix", "mini"}
)
_PAL_TOKEN = re.compile(r"^pal(\d+)$", re.IGNORECASE)
_PAL_BLOCK = re.compile(r"%pal\b(.*?)\bend\b", re.IGNORECASE | re.DOTALL)
_PAL_NPROCS = re.compile(r"\bnprocs\s+(\d+)", re.IGNORECASE)
_MAXCORE = re.compile(r"%maxcore\s+(\S+)", re.IGNORECASE)


@dataclass
class ParsedOrcaInput:
    """orcasimpleinput, sorted into what the keywords mean."""

    method: str | None = None
    method_class: str | None = None
    cost_class: str | None = None
    rung: str | None = None
    numerical_gradients: bool = False
    basis: str | None = None
    dispersion: str = "none"
    relativity: str = "none"
    spin_treatment: str = "per_structure"
    solvation: str = "none"
    approximations: list[str] = field(default_factory=list)
    scf_threshold: str | None = None
    grid: str | None = None
    pal: list[int] = field(default_factory=list)


def _wavefunction_family(token: str) -> str:
    """``dlpno-ccsd(t)`` -> ``ccsd(t)``; ``ri-mp2-f12`` -> ``mp2``."""
    family = token
    changed = True
    while changed:
        changed = False
        for prefix in _WAVEFUNCTION_PREFIXES:
            if family.startswith(prefix) and len(family) > len(prefix):
                family = family[len(prefix) :]
                changed = True
    return re.sub(r"-?f12$", "", family)


def _is_basis(token: str) -> bool:
    if "/" in token:
        return False  # auxiliary basis (def2/J, cc-pVTZ/C)
    return token in _BASIS_NAMES or token.startswith(_BASIS_PREFIXES)


def parse_orca_simple_input(simpleinput: str, orcablocks: str = "") -> ParsedOrcaInput:
    """Sort the keywords of an ORCA simple input line (``!`` optional,
    case-insensitive) into method, basis, dispersion, relativity,
    solvation, approximations and numerical settings. Keywords it doesn't
    know go into ``approximations``, so they still distinguish one level
    of theory from another."""
    parsed = ParsedOrcaInput()
    unknown: list[str] = []
    for raw in simpleinput.replace("!", " ").split():
        token = raw.lower()
        if (pal := _PAL_TOKEN.match(token)) is not None:
            parsed.pal.append(int(pal.group(1)))
        elif token in _RUN_KEYWORDS:
            continue
        elif token in _SCF_THRESHOLDS:
            parsed.scf_threshold = raw
        elif _GRID.match(token):
            parsed.grid = raw
        elif token in _RI_APPROXIMATIONS or "/" in token:
            parsed.approximations.append(token)
        elif token in _RELATIVITY:
            parsed.relativity = _RELATIVITY[token]
        elif token in _DISPERSION:
            parsed.dispersion = _DISPERSION[token]
        elif token in _SPIN:
            parsed.spin_treatment = _SPIN[token]
        elif _SOLVATION.match(token):
            parsed.solvation = raw
        elif _is_basis(token):
            parsed.basis = raw
        elif parsed.method is None and _set_method(parsed, raw, token):
            continue
        else:
            unknown.append(token)
    if any(block in orcablocks.lower() for block in _MULTIREFERENCE_BLOCKS):
        parsed.cost_class = MULTIREFERENCE
        parsed.method_class = MULTIREFERENCE
        parsed.numerical_gradients = True
    parsed.approximations = sorted(set(parsed.approximations + unknown))
    return parsed


def _set_method(parsed: ParsedOrcaInput, raw: str, token: str) -> bool:
    """Record *token* as the method if it names one; False if it doesn't."""
    # An active space argument (CASSCF(6,6)) isn't part of the family.
    family = re.sub(r"\(\d+,\s*\d+\)$", "", _wavefunction_family(token))
    if family in _CORRELATED_FAMILIES:
        parsed.method, parsed.method_class = raw, CORRELATED
        parsed.cost_class = CORRELATED
        parsed.numerical_gradients = family in _NUMERICAL_GRADIENT_FAMILIES
        return True
    if family in _MULTIREFERENCE_FAMILIES:
        parsed.method, parsed.method_class = raw, MULTIREFERENCE
        parsed.cost_class = MULTIREFERENCE
        parsed.numerical_gradients = family in _NUMERICAL_GRADIENT_FAMILIES
        return True
    if token in _SEMI_EMPIRICAL:
        parsed.method, parsed.method_class = raw, SEMI_EMPIRICAL
        parsed.cost_class = SEMI_EMPIRICAL
        return True
    if is_hartree_fock(token):
        parsed.method, parsed.method_class = "HF", "mean_field_hf"
        parsed.cost_class = MEAN_FIELD
        return True
    composite = _COMPOSITE_3C.match(token)
    if composite is not None:
        parsed.method, parsed.method_class = raw, "dft"
        parsed.cost_class = MEAN_FIELD
        base = composite.group(1)
        if is_hartree_fock(base):
            parsed.method_class = "mean_field_hf"
        else:
            parsed.rung = functional_rung(base)[1]
        parsed.basis = "built-in (3c)"
        parsed.dispersion = "built-in (3c)"
        return True
    name, rung = functional_rung(raw)
    if rung is None:
        return False
    parsed.method, parsed.method_class, parsed.rung = name, "dft", rung
    # Double hybrids add an MP2 step: they cost what MP2 costs.
    parsed.cost_class = CORRELATED if rung == "double-hybrid" else MEAN_FIELD
    stripped = re.search(r"-(d3bj|d3zero|d3|d4|d2)$", token)
    if stripped is not None and _normalise_name(name) != token:
        # A dispersion suffix on a plain functional (PBE-D3BJ).
        parsed.dispersion = _DISPERSION[stripped.group(1)]
    elif re.search(r"-(d|d3|d4|v)$", name.lower()):
        # Part of the functional's own definition (wB97X-D3, B97-D, wB97M-V).
        parsed.dispersion = "in functional"
    return True


def _normalise_name(name: str | None) -> str:
    return re.sub(r"[\s_]", "", name or "").lower()


def classify_orca_method(simpleinput: str, orcablocks: str = "") -> str | None:
    """The cost class of an ORCA method: ``semi_empirical``,
    ``mean_field`` (HF, DFT), ``correlated`` (MP2, coupled cluster,
    double hybrids) or ``multireference`` (CASSCF, NEVPT2, MRCI); None
    when no method keyword is recognised."""
    return parse_orca_simple_input(simpleinput, orcablocks).cost_class


def resolve_orca_kwargs(orca_kwargs: dict | None) -> dict:
    """ORCA_KWARGS_DEFAULTS with the user's orca_kwargs over them."""
    return {**ORCA_KWARGS_DEFAULTS, **(orca_kwargs or {})}


def _single_value(name: str, values: list[int | None]) -> int | None:
    """The one value *name* was given (None if none); raise when it was
    given more than once with different values."""
    distinct = sorted({v for v in values if v is not None})
    if len(distinct) > 1:
        raise ValueError(
            f"{_WHERE}: {name} is set more than once with different values "
            f"{distinct} (orca_kwargs.{name}, a %pal/%maxcore block in "
            "orcablocks, or a PALn keyword). Set it in one place."
        )
    return distinct[0] if distinct else None


def _int_setting(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{_WHERE}: {name} must be an integer, got {value!r}.")


@dataclass
class _OrcaInputParts:
    keywords: list[str]  # simple-input keywords without PALn
    blocks: str  # orcablocks without %pal/%maxcore
    nprocs: int | None  # user-set, if any
    maxcore: int | None


def _split_user_input(kwargs: dict) -> _OrcaInputParts:
    """Take the user's parallel settings out of orcasimpleinput and
    orcablocks, checking that each is set at most once."""
    simpleinput = kwargs["orcasimpleinput"]
    if not isinstance(simpleinput, str) or not simpleinput.strip():
        raise ValueError(
            f"{_WHERE}.orcasimpleinput is required: the ORCA keyword line, "
            "e.g. 'RI-MP2 def2-TZVP def2-TZVP/C' or 'r2SCAN-3c'."
        )
    blocks = kwargs["orcablocks"] or ""
    if not isinstance(blocks, str):
        raise ValueError(f"{_WHERE}.orcablocks must be a string, got {blocks!r}.")
    tokens = simpleinput.replace("!", " ").split()
    nprocs = _single_value(
        "nprocs",
        [
            _int_setting("nprocs", kwargs["nprocs"]),
            *(
                int(m.group(1))
                for block in _PAL_BLOCK.findall(blocks)
                if (m := _PAL_NPROCS.search(block)) is not None
            ),
            *(int(m.group(1)) for t in tokens if (m := _PAL_TOKEN.match(t))),
        ],
    )
    maxcore = _single_value(
        "maxcore",
        [
            _int_setting("maxcore", kwargs["maxcore"]),
            *(_int_setting("%maxcore", v) for v in _MAXCORE.findall(blocks)),
        ],
    )
    return _OrcaInputParts(
        keywords=[t for t in tokens if not _PAL_TOKEN.match(t)],
        blocks=_MAXCORE.sub("", _PAL_BLOCK.sub("", blocks)).strip(),
        nprocs=nprocs,
        maxcore=maxcore,
    )


def build_orca_input(orca_kwargs: dict | None, node_info: dict) -> dict[str, str]:
    """The orcasimpleinput and orcablocks ORCA is run with.

    The user's ``%pal``/``%maxcore`` blocks and ``PALn`` keywords are taken
    out and written once, with the values from resolve_parallel_resources
    (the user's if given, else the HPC profile's); EnGrad and
    SCFConvergenceForced are added when missing."""
    parts = _split_user_input(resolve_orca_kwargs(orca_kwargs))
    resources = resolve_parallel_resources(
        node_info,
        nprocs=parts.nprocs,
        mem_per_proc_mb=parts.maxcore,
        where=_WHERE,
    )
    lower = {t.lower() for t in parts.keywords}
    keywords = parts.keywords + [
        k for k in _REQUIRED_KEYWORDS if k.lower() not in lower
    ]
    resource_blocks = (
        f"%pal nprocs {resources.nprocs} end\n%maxcore {resources.mem_per_proc_mb}"
    )
    return {
        "orcasimpleinput": " ".join(keywords),
        "orcablocks": "\n".join(b for b in (resource_blocks, parts.blocks) if b),
    }


def _as_int(value: object) -> int | None:
    """*value* as a Python int when it is a whole number of any numeric
    type: extxyz reads info integers back as numpy integers
    (np.int64), and floats like 1.0 are accepted too. None otherwise
    (bools, strings, fractional values)."""
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return int(value)
    return None


def structure_charge_and_multiplicity(
    atoms: Atoms, default_charge: int = 0
) -> tuple[int, int]:
    """Charge and spin multiplicity for one structure.

    Charge: ``atoms.info["charge"]``, else *default_charge*. Multiplicity:
    ``atoms.info["spin_multiplicity"]``; else for a neutral single atom its
    ground state's (Hund's rule, ase.data.ground_state_magnetic_moments +
    1: H 2, C 3, O 3); else the lowest the electron count allows (1 when
    even, 2 when odd). Raises when the two can't go together."""
    charge = _as_int(atoms.info.get("charge", default_charge))
    if charge is None:
        raise ValueError(
            f"ORCA needs an integer charge, got {atoms.info.get('charge')!r}."
        )
    n_electrons = int(atoms.numbers.sum()) - charge
    if n_electrons < 0:
        raise ValueError(
            f"Charge {charge} leaves {n_electrons} electrons in "
            f"{atoms.get_chemical_formula()}."
        )
    mult = atoms.info.get("spin_multiplicity")
    if mult is None:
        if len(atoms) == 1 and charge == 0:
            mult = round(ground_state_magnetic_moments[atoms.numbers[0]]) + 1
        else:
            mult = 1 if n_electrons % 2 == 0 else 2
    mult = _as_int(mult) if _as_int(mult) is not None else mult
    if (
        not isinstance(mult, int)
        or isinstance(mult, bool)
        or mult < 1
        or mult - 1 > n_electrons
        or (n_electrons + mult - 1) % 2 != 0
    ):
        raise ValueError(
            f"Spin multiplicity {mult!r} is impossible for "
            f"{atoms.get_chemical_formula()} with charge {charge} "
            f"({n_electrons} electrons: an {'even' if n_electrons % 2 == 0 else 'odd'}"
            " count needs an "
            f"{'odd' if n_electrons % 2 == 0 else 'even'} multiplicity). Fix "
            "atoms.info['spin_multiplicity'] or atoms.info['charge']."
        )
    return charge, mult


def create_orca_calc_object(
    atoms: Atoms, high_accuracy_eval_job_dict: dict, out_dir: str
) -> ORCA:
    """An ASE ORCA calculator for *atoms* (the stock interface: input from
    build_orca_input, charge and multiplicity per structure)."""
    if atoms.pbc.any():
        raise ValueError(
            f"ORCA has no periodic boundary conditions, but structure "
            f"{atoms.get_chemical_formula()} (config_type "
            f"{atoms.info.get('config_type', 'unknown')!r}) has pbc "
            f"{atoms.pbc.tolist()}. Only non-periodic structures (pbc all "
            "False) can go to ORCA."
        )
    hpc = high_accuracy_eval_job_dict["hpc"]
    if not hpc.get("orca_path"):
        raise ValueError(
            f"The HPC profile {hpc.get('hpc_name')!r} has no orca_path (the "
            "absolute path of the orca binary)."
        )
    orca_kwargs = resolve_orca_kwargs(high_accuracy_eval_job_dict.get("orca_kwargs"))
    orca_input = build_orca_input(orca_kwargs, hpc["node_info"])
    charge, mult = structure_charge_and_multiplicity(atoms, orca_kwargs["charge"])
    return ORCA(
        profile=OrcaProfile(command=hpc["orca_path"]),
        directory=out_dir,
        charge=charge,
        mult=mult,
        **orca_input,
    )


def run_sp_orca(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_sp(
        input_structure, out_dir, high_accuracy_eval_job_dict, create_orca_calc_object
    )


def run_go_orca(
    input_structure: Atoms,
    out_dir: str,
    high_accuracy_eval_job_dict: dict,
) -> Atoms:
    return _run_go(
        input_structure,
        out_dir,
        high_accuracy_eval_job_dict,
        create_orca_calc_object,
        opt_prefix="orca_opt",
    )


# -- driver-side entry points (registry) -------------------------------------


def resolve_effective_kwargs(orca_kwargs: dict) -> dict:
    """For the pre-run summary: the defaults merged in and the keywords
    build_orca_input adds. nprocs/maxcore show None when they come from
    the HPC profile (resolved per job)."""
    effective = resolve_orca_kwargs(orca_kwargs)
    simpleinput = effective["orcasimpleinput"]
    if isinstance(simpleinput, str):
        lower = {t.lower() for t in simpleinput.replace("!", " ").split()}
        effective["orcasimpleinput"] = " ".join(
            [simpleinput.strip()]
            + [k for k in _REQUIRED_KEYWORDS if k.lower() not in lower]
        )
    return effective


def max_num_of_atoms(config: dict) -> int | None:
    """The largest structure that may go to ORCA: the cap
    (orca_kwargs.max_num_of_atoms_correlated) for correlated and
    multireference methods, None (no limit) otherwise."""
    kwargs = resolve_orca_kwargs(config.get("orca_kwargs"))
    simpleinput = kwargs["orcasimpleinput"]
    if not isinstance(simpleinput, str):
        return None
    cost = classify_orca_method(simpleinput, kwargs["orcablocks"] or "")
    if cost not in _CAPPED_COST_CLASSES:
        return None
    cap = kwargs["max_num_of_atoms_correlated"]
    return None if cap is None else int(cap)


def level_of_theory(config: dict, hpc: dict | None = None) -> LevelOfTheory:  # noqa: ARG001
    kwargs = resolve_orca_kwargs(config.get("orca_kwargs"))
    parsed = parse_orca_simple_input(
        kwargs["orcasimpleinput"] or "", kwargs["orcablocks"] or ""
    )
    pseudopotentials = None
    core = "all_electron"
    basis = (parsed.basis or "").lower()
    if basis.startswith(("def2-", "ma-def2")) and parsed.relativity == "none":
        # def2 bases come with ECPs for elements beyond Kr.
        core, pseudopotentials = "ecp", {"Z>36": "def2-ECP"}
    numerical = {"grid": parsed.grid} if parsed.grid else {}
    return LevelOfTheory(
        code="orca",
        method_class=parsed.method_class,
        method=parsed.method,
        rung=parsed.rung,
        dispersion=parsed.dispersion,
        hubbard_u={},
        basis_type="gaussian",
        basis=parsed.basis,
        core_treatment=core,
        pseudopotentials=pseudopotentials,
        relativity=parsed.relativity,
        spin_treatment=parsed.spin_treatment,
        solvation=parsed.solvation,
        approximations=parsed.approximations,
        scf_threshold=parsed.scf_threshold,
        numerical_settings=numerical,
    )


def validate_kwargs(config: dict, hpc: dict | None) -> None:
    """Construction-time check: orca_kwargs readable and complete, the
    parallel settings fit the HPC profile, the cap valid. Warns when the
    method isn't recognised (no atom cap applied) or has no analytic
    gradients."""
    orca_kwargs = config.get("orca_kwargs") or {}
    if not isinstance(orca_kwargs, dict):
        raise ValueError(f"{_WHERE} must be a mapping, got {orca_kwargs!r}.")
    unknown = sorted(set(orca_kwargs) - set(ORCA_KWARGS_DEFAULTS))
    if unknown:
        raise ValueError(
            f"{_WHERE}: unknown keys {unknown}; expected "
            f"{sorted(ORCA_KWARGS_DEFAULTS)}. ORCA settings go in "
            "orcasimpleinput (keywords) or orcablocks (% blocks)."
        )
    kwargs = resolve_orca_kwargs(orca_kwargs)
    cap = kwargs["max_num_of_atoms_correlated"]
    if cap is not None and (
        not isinstance(cap, int) or isinstance(cap, bool) or cap < 1
    ):
        raise ValueError(
            f"{_WHERE}.max_num_of_atoms_correlated must be a positive integer "
            f"or null (no cap), got {cap!r}."
        )
    charge = kwargs["charge"]
    if not isinstance(charge, int) or isinstance(charge, bool):
        raise ValueError(f"{_WHERE}.charge must be an integer, got {charge!r}.")
    if isinstance(hpc, dict):
        if not hpc.get("orca_path"):
            raise ValueError(
                f"The HPC profile {hpc.get('hpc_name')!r} used by "
                "high_accuracy_evaluation has no orca_path: add the absolute "
                "path of the orca binary to it in ~/.alomancy/hpc_config.yaml "
                "(ORCA must be started by its full path to run in parallel), "
                "and load OpenMPI in its pre_cmds."
            )
        node_info = hpc.get("node_info") or {}
        build_orca_input(kwargs, node_info)  # raises if the resources don't fit
        if node_info.get("threads_per_rank", 1) != 1:
            logger.warning(
                "The HPC profile %r sets node_info.threads_per_rank %s; ORCA "
                "parallelises with MPI processes only (%%pal nprocs), so the "
                "extra threads per process stay idle.",
                hpc.get("hpc_name"),
                node_info.get("threads_per_rank"),
            )
    else:
        _split_user_input(kwargs)  # no profile to check the resources against

    parsed = parse_orca_simple_input(
        kwargs["orcasimpleinput"], kwargs["orcablocks"] or ""
    )
    if parsed.cost_class is None:
        logger.warning(
            "Couldn't recognise the method in %s.orcasimpleinput %r: the "
            "level of theory records no method, and the atom cap "
            "(max_num_of_atoms_correlated) for correlated methods is not "
            "applied. If it is a correlated method, raise an issue so it "
            "can be added to the list.",
            _WHERE,
            kwargs["orcasimpleinput"],
            extra={
                "event": "orca_method_unrecognised",
                "data": {"orcasimpleinput": kwargs["orcasimpleinput"]},
            },
        )
    if parsed.numerical_gradients:
        logger.warning(
            "%s (%s) may have no analytic gradients in ORCA: forces would "
            "come from finite differences, about 6 x the number of atoms "
            "energy calculations per structure. Check the ORCA manual for "
            "your version (it may need the NumGrad keyword), and keep "
            "max_num_of_atoms_correlated (%s) small.",
            parsed.method or "the configured multireference method",
            kwargs["orcasimpleinput"],
            cap,
            extra={
                "event": "numerical_gradients",
                "data": {"method": parsed.method, "cap": cap},
            },
        )


def report_section(stats: dict, **kwargs: Any) -> Any:
    """Loop-report section for this module (see analysis/report/sections.py)."""
    from alomancy.analysis.report.sections import dft_section

    return dft_section(stats, **kwargs)
