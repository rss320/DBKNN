"""APT dataset generation functions (lazy ase import)."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def _import_ase() -> tuple:
    """Lazily import ase modules."""
    try:
        from ase import Atoms
        from ase.build import bulk
        from ase.io import write
    except ImportError as exc:
        raise ImportError("Install dbknn[apt] for structure generation") from exc
    return Atoms, bulk, write


def create_bcc_supercell(n_cells: int = 30, lattice_param: float = 2.866, element: str = "Fe"):  # noqa: ANN201
    """Build a BCC supercell."""
    _, bulk, _ = _import_ase()
    unit = bulk(element, "bcc", a=lattice_param, cubic=True)
    atoms = unit.repeat((n_cells, n_cells, n_cells))
    log.info("Created BCC supercell: %d atoms, box=%.1f \u00c5", len(atoms), atoms.cell[0, 0])
    return atoms


def assign_composition(
    atoms: object,
    composition: Optional[dict[str, float]] = None,
    matrix_element: str = "Fe",
    rng: Optional[np.random.Generator] = None,
) -> object:
    """Randomly assign solute species to matrix sites by composition (at%)."""
    if composition is None:
        composition = {"Ni": 0.0081, "Mn": 0.0162, "Cu": 0.0007}
    if rng is None:
        rng = np.random.default_rng()

    n = len(atoms)  # type: ignore[arg-type]
    symbols = np.array([matrix_element] * n)
    indices = rng.permutation(n)
    offset = 0
    for species, frac in composition.items():
        count = int(round(frac * n))
        symbols[indices[offset : offset + count]] = species
        offset += count
        log.info("Assigned %d %s atoms (%.2f%%)", count, species, frac * 100)

    atoms.set_chemical_symbols(symbols.tolist())  # type: ignore[attr-defined]
    return atoms


def insert_clusters(
    atoms: object,
    n_clusters: int = 5,
    cluster_size: int = 20,
    relative_density: float = 200.0,
    composition: Optional[dict[str, float]] = None,
    matrix_element: str = "Fe",
    rng: Optional[np.random.Generator] = None,
) -> object:
    """Insert solute clusters at random positions in the supercell."""
    Atoms, _, _ = _import_ase()

    if composition is None:
        composition = {"Ni": 0.0081, "Mn": 0.0162, "Cu": 0.0007}
    if rng is None:
        rng = np.random.default_rng()

    positions = atoms.get_positions()  # type: ignore[attr-defined]
    cell = atoms.cell.array  # type: ignore[attr-defined]
    box_lengths = np.diag(cell)
    symbols = np.array(atoms.get_chemical_symbols())  # type: ignore[attr-defined]

    matrix_density = len(atoms) / np.prod(box_lengths)  # type: ignore[arg-type]
    cluster_density = relative_density * matrix_density

    radius = (3 * cluster_size / (4 * np.pi * cluster_density)) ** (1.0 / 3.0)
    sigma = radius / 2.0
    log.info(
        "Cluster params: radius=%.2f \u00c5, sigma=%.2f \u00c5, density=%.4f atoms/\u00c5\u00b3",
        radius, sigma, cluster_density,
    )

    cluster_labels = np.zeros(len(atoms), dtype=int)  # type: ignore[arg-type]

    total_frac = sum(composition.values())
    species_list = list(composition.keys())
    cum_fracs = np.cumsum([composition[s] / total_frac for s in species_list])

    for ci in range(n_clusters):
        center = rng.uniform(low=radius * 2, high=box_lengths - radius * 2)

        dists: NDArray[np.floating] = np.linalg.norm(positions - center, axis=1)
        in_region = dists < 2 * radius
        solute_mask = np.isin(symbols, species_list)
        to_revert = in_region & solute_mask
        symbols[to_revert] = matrix_element
        log.info("Cluster %d: reverted %d solutes near center", ci + 1, int(to_revert.sum()))

        cluster_pos = rng.normal(loc=center, scale=sigma, size=(cluster_size, 3))
        cluster_pos = np.mod(cluster_pos, box_lengths)

        r = rng.random(cluster_size)
        cluster_symbols = []
        for val in r:
            idx = int(np.searchsorted(cum_fracs, val))
            cluster_symbols.append(species_list[min(idx, len(species_list) - 1)])

        new_positions = np.vstack([positions, cluster_pos])
        new_symbols = np.concatenate([symbols, np.array(cluster_symbols)])
        cluster_labels = np.concatenate([
            cluster_labels, np.full(cluster_size, ci + 1, dtype=int)
        ])
        positions = new_positions
        symbols = new_symbols
        log.info("Cluster %d: inserted %d atoms at (%.1f, %.1f, %.1f)", ci + 1, cluster_size, *center)

    result = Atoms(
        symbols=symbols.tolist(),
        positions=positions,
        cell=cell,
        pbc=True,
    )
    result.arrays["cluster_label"] = cluster_labels
    log.info("Total atoms after cluster insertion: %d", len(result))
    return result


def apply_detector_efficiency(
    atoms: object,
    efficiency: float = 0.55,
    rng: Optional[np.random.Generator] = None,
) -> object:
    """Remove atoms to simulate detector efficiency."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(atoms)  # type: ignore[arg-type]
    keep = rng.random(n) < efficiency
    new_atoms = atoms[keep]  # type: ignore[index]
    new_atoms.arrays["cluster_label"] = atoms.arrays["cluster_label"][keep]  # type: ignore[attr-defined]
    log.info("Detector efficiency %.0f%%: %d \u2192 %d atoms", efficiency * 100, n, len(new_atoms))
    return new_atoms


def apply_positional_noise(
    atoms: object,
    noise: tuple[float, float, float] = (5.0, 5.0, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> object:
    """Add Gaussian positional noise (in angstroms)."""
    if rng is None:
        rng = np.random.default_rng()

    positions = atoms.get_positions()  # type: ignore[attr-defined]
    noise_arr = rng.normal(0, noise, size=positions.shape)
    atoms.set_positions(positions + noise_arr)  # type: ignore[attr-defined]
    log.info("Applied positional noise: (%.1f, %.1f, %.1f) \u00c5", *noise)
    return atoms


def generate_apt_dataset(
    n_cells: int = 30,
    n_clusters: int = 5,
    cluster_size: int = 20,
    relative_density: float = 200.0,
    efficiency: float = 0.55,
    noise: tuple[float, float, float] = (5.0, 5.0, 1.0),
    composition: Optional[dict[str, float]] = None,
    matrix_element: str = "Fe",
    lattice_param: float = 2.866,
    seed: int = 42,
    output: str = "simulated_apt.xyz",
) -> object:
    """Build supercell, assign composition, insert clusters, apply efficiency + noise, write."""
    _, _, write = _import_ase()
    rng = np.random.default_rng(seed)

    atoms = create_bcc_supercell(n_cells=n_cells, lattice_param=lattice_param, element=matrix_element)
    atoms = assign_composition(atoms, composition=composition, matrix_element=matrix_element, rng=rng)
    atoms = insert_clusters(
        atoms, n_clusters=n_clusters, cluster_size=cluster_size,
        relative_density=relative_density, composition=composition,
        matrix_element=matrix_element, rng=rng,
    )
    atoms = apply_detector_efficiency(atoms, efficiency=efficiency, rng=rng)
    atoms = apply_positional_noise(atoms, noise=noise, rng=rng)

    write(output, atoms, format="extxyz")
    log.info("Wrote %s (%d atoms)", output, len(atoms))
    return atoms
