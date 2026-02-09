"""I/O functions for APT data."""

import csv
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

NM_TO_ANGSTROM: float = 10.0

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def _solute_mask(
    symbols: NDArray[np.str_],
    solute_species: Optional[list[str]] = None,
    matrix_species: Optional[list[str]] = None,
) -> tuple[NDArray[np.bool_], list[str]]:
    """Build a boolean solute mask from either solute or matrix species.

    Exactly one of ``solute_species`` / ``matrix_species`` should be given.
    If neither is provided, defaults to ``solute_species=["Ni", "Mn", "Cu"]``.

    Returns ``(mask, resolved_solute_list)`` for logging.
    """
    if solute_species is not None and matrix_species is not None:
        raise ValueError("Specify solute_species or matrix_species, not both")

    if matrix_species is not None:
        mask: NDArray[np.bool_] = ~np.isin(symbols, matrix_species)
        resolved = sorted(set(symbols[mask].tolist()))
    else:
        if solute_species is None:
            solute_species = ["Ni", "Mn", "Cu"]
        mask = np.isin(symbols, solute_species)
        resolved = list(solute_species)

    return mask, resolved


def mc_to_species(
    mc: NDArray[np.floating],
    mc_ranges: dict[str, list[tuple[float, float]]],
    unranged_label: str = "unranged",
) -> NDArray[np.str_]:
    """Convert mass-to-charge ratios to species labels.

    Parameters
    ----------
    mc : array of float
        Mass-to-charge ratios.
    mc_ranges : dict mapping species to list of (low, high) intervals
        E.g. ``{"Fe": [(18.03, 19.02), (22.12, 23.89)], "Ni": [(9.8, 10.5)]}``.
    unranged_label : str
        Label assigned to ions that fall outside all ranges.

    Returns
    -------
    NDArray[np.str_]
        Species label for each ion.
    """
    species = np.full(len(mc), unranged_label, dtype="U20")
    for label, ranges in mc_ranges.items():
        for low, high in ranges:
            mask = (mc >= low) & (mc <= high)
            species[mask] = label
    n_unranged = int(np.sum(species == unranged_label))
    if n_unranged > 0:
        log.info("%d / %d ions unranged (%.1f%%)", n_unranged, len(mc), 100.0 * n_unranged / len(mc))
    return species


def load_apt_csv(
    filepath: str,
    mc_ranges: dict[str, list[tuple[float, float]]],
    solute_species: Optional[list[str]] = None,
    matrix_species: Optional[list[str]] = None,
    unranged_label: str = "unranged",
    skiprows: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], None]:
    """Load APT data from CSV with columns ``[x, y, z, mc]``.

    Positions are converted from **nm to angstrom** (×10).
    Mass-to-charge ratios are mapped to species via *mc_ranges*.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    mc_ranges : dict mapping species to list of (low, high) intervals
        E.g. ``{"Fe": [(18.03, 19.02)], "Cu": [(31.0, 32.5)]}``
    solute_species / matrix_species : list[str] | None
        Same semantics as :func:`load_apt_data`.
    unranged_label : str
        Species label for ions outside all *mc_ranges*.
    skiprows : int
        Number of header rows to skip (default 0).

    Returns
    -------
    (solute_positions, all_positions, None)
        Same shape as :func:`load_apt_data`; ground-truth is always *None*.
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Expected >= 4 columns [x, y, z, mc], got {data.shape[1]}")

    positions: NDArray[np.floating] = data[:, :3] * NM_TO_ANGSTROM
    mc: NDArray[np.floating] = data[:, 3]

    symbols = mc_to_species(mc, mc_ranges, unranged_label=unranged_label)

    solute_mask, resolved = _solute_mask(symbols, solute_species, matrix_species)
    solute_positions = positions[solute_mask]

    log.info(
        "Loaded %s: %d total ions, %d solutes (%s)",
        filepath, len(data), len(solute_positions), ", ".join(resolved),
    )
    return solute_positions, positions, None


def save_labeled_csv(
    filepath: str,
    output: str,
    mc_ranges: dict[str, list[tuple[float, float]]],
    solute_species: Optional[list[str]] = None,
    matrix_species: Optional[list[str]] = None,
    solute_labels: Optional[NDArray[np.integer]] = None,
    solute_scores: Optional[NDArray[np.floating]] = None,
    unranged_label: str = "unranged",
    skiprows: int = 0,
) -> None:
    """Write CSV with appended ``species``, ``cluster_label``, and ``hybrid_score`` columns.

    Non-solute ions get label=0 and score=NaN.
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=skiprows)
    positions_nm = data[:, :3]
    mc: NDArray[np.floating] = data[:, 3]
    symbols = mc_to_species(mc, mc_ranges, unranged_label=unranged_label)

    solute_mask, _ = _solute_mask(symbols, solute_species, matrix_species)

    all_labels = np.zeros(len(data), dtype=int)
    all_scores = np.full(len(data), np.nan, dtype=np.float64)
    if solute_labels is not None:
        all_labels[solute_mask] = solute_labels
    if solute_scores is not None:
        all_scores[solute_mask] = solute_scores

    out = Path(output)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_nm", "y_nm", "z_nm", "mc", "species", "cluster_label", "hybrid_score"])
        for i in range(len(data)):
            writer.writerow([
                positions_nm[i, 0], positions_nm[i, 1], positions_nm[i, 2],
                mc[i], symbols[i], all_labels[i], all_scores[i],
            ])

    n_cluster = int(all_labels.sum())
    log.info("Saved %s: %d cluster atoms / %d total", output, n_cluster, len(data))


def load_apt_data(
    filepath: str,
    solute_species: Optional[list[str]] = None,
    matrix_species: Optional[list[str]] = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Optional[NDArray[np.integer]]]:
    """Load APT data, return (solute_positions, all_positions, ground_truth_labels or None).

    Specify either ``solute_species`` (include list) or ``matrix_species``
    (exclude list — all other species are treated as solutes).
    """
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("Install dbknn[apt] for APT file support") from exc

    atoms = read(filepath, format="extxyz")
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    solute_mask, resolved = _solute_mask(symbols, solute_species, matrix_species)
    solute_positions = positions[solute_mask]

    ground_truth: Optional[NDArray[np.integer]] = None
    if "cluster_label" in atoms.arrays:
        labels = atoms.arrays["cluster_label"][solute_mask]
        ground_truth = (labels > 0).astype(int)

    log.info(
        "Loaded %s: %d total atoms, %d solutes (%s)",
        filepath, len(atoms), len(solute_positions), ", ".join(resolved),
    )
    return solute_positions, positions, ground_truth


def save_labeled_xyz(
    filepath: str,
    output: str,
    solute_species: Optional[list[str]] = None,
    matrix_species: Optional[list[str]] = None,
    solute_labels: Optional[NDArray[np.integer]] = None,
    solute_scores: Optional[NDArray[np.floating]] = None,
) -> None:
    """Write extended XYZ with cluster_label and hybrid_score on each atom.

    Non-solute atoms get label=0 and score=NaN.
    """
    try:
        from ase.io import read, write
    except ImportError as exc:
        raise ImportError("Install dbknn[apt] for APT file support") from exc

    atoms = read(filepath, format="extxyz")
    symbols = np.array(atoms.get_chemical_symbols())
    solute_mask, _ = _solute_mask(symbols, solute_species, matrix_species)

    # Map solute-only arrays back to all atoms
    all_labels = np.zeros(len(atoms), dtype=int)
    all_scores = np.full(len(atoms), np.nan, dtype=np.float64)

    if solute_labels is not None:
        all_labels[solute_mask] = solute_labels
    if solute_scores is not None:
        all_scores[solute_mask] = solute_scores

    atoms.arrays["cluster_label"] = all_labels
    atoms.arrays["hybrid_score"] = all_scores

    write(output, atoms, format="extxyz")
    n_cluster = int(all_labels.sum())
    log.info("Saved %s: %d cluster atoms / %d total", output, n_cluster, len(atoms))


def _species_to_element(species: str) -> str:
    """Extract the leading element symbol from a ranged species label.

    ``"Fe"`` → ``"Fe"``, ``"FeO"`` → ``"Fe"``, ``"C2"`` → ``"C"``,
    ``"unranged"`` → ``"X"`` (dummy).
    """
    m = _ELEMENT_RE.match(species)
    return m.group(1) if m else "X"


def save_csv_as_xyz(
    filepath: str,
    output: str,
    mc_ranges: dict[str, list[tuple[float, float]]],
    solute_species: Optional[list[str]] = None,
    matrix_species: Optional[list[str]] = None,
    solute_labels: Optional[NDArray[np.integer]] = None,
    solute_scores: Optional[NDArray[np.floating]] = None,
    unranged_label: str = "unranged",
    skiprows: int = 0,
) -> None:
    """Convert an APT CSV to extended XYZ with cluster labels and scores.

    Positions are converted from nm to angstrom.  Each atom stores:

    * ``species`` — full ranged label (e.g. ``"FeO"``, ``"Ni"``)
    * ``cluster_label`` — 1 = cluster, 0 = matrix (non-solutes get 0)
    * ``hybrid_score`` — DBKNN hybrid score (non-solutes get NaN)

    Molecular-ion labels are mapped to their primary element for the
    XYZ species column (e.g. ``FeO`` → ``Fe``).
    """
    try:
        from ase import Atoms
        from ase.io import write
    except ImportError as exc:
        raise ImportError("Install dbknn[apt] for XYZ file support") from exc

    data = np.loadtxt(filepath, delimiter=",", skiprows=skiprows)
    positions_ang = data[:, :3] * NM_TO_ANGSTROM
    mc: NDArray[np.floating] = data[:, 3]
    species = mc_to_species(mc, mc_ranges, unranged_label=unranged_label)

    solute_mask, _ = _solute_mask(species, solute_species, matrix_species)

    all_labels = np.zeros(len(data), dtype=int)
    all_scores = np.full(len(data), np.nan, dtype=np.float64)
    if solute_labels is not None:
        all_labels[solute_mask] = solute_labels
    if solute_scores is not None:
        all_scores[solute_mask] = solute_scores

    elements = [_species_to_element(s) for s in species]
    atoms = Atoms(symbols=elements, positions=positions_ang)
    atoms.arrays["ranged_species"] = species
    atoms.arrays["cluster_label"] = all_labels
    atoms.arrays["hybrid_score"] = all_scores

    write(output, atoms, format="extxyz")
    n_cluster = int(all_labels.sum())
    log.info("Saved %s: %d cluster atoms / %d total", output, n_cluster, len(atoms))
