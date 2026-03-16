"""TSP utilities: TSPLib parsing, distance matrix, and instance generation."""
import pathlib
import numpy as np


class TSPInstance:
    """Hashable TSP instance for use as ProgramsDatabase test input key."""

    def __init__(self, instance_id: str, dist_matrix: np.ndarray):
        self.instance_id = instance_id
        self.dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

    def __hash__(self) -> int:
        return hash(self.instance_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TSPInstance):
            return self.instance_id == other.instance_id
        return False

    def __lt__(self, other: "TSPInstance") -> bool:
        return self.instance_id < other.instance_id


def coords_to_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix from 2D coordinates.

    Args:
        coords: Array of shape (n, 2) with (x, y) coordinates.

    Returns:
        Distance matrix of shape (n, n).
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def load_tsplib(filepath: str | pathlib.Path) -> tuple[str, np.ndarray]:
    """Parse TSPLib file (EUC_2D format) and return (name, coords).

    Args:
        filepath: Path to .tsp file.

    Returns:
        Tuple of (instance_name, coords) where coords is (n, 2) array.
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"TSPLib file not found: {filepath}")

    name = path.stem
    coords = []
    in_section = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if in_section:
                if line in ("EOF", ""):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])

    coords = np.array(coords, dtype=np.float64)
    return name, coords


def generate_random_euclidean_tsp(
    n: int, seed: int | None = None
) -> tuple[str, np.ndarray]:
    """Generate random Euclidean TSP instance in unit square.

    Args:
        n: Number of cities.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (instance_id, coords) where coords is (n, 2) array.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1, size=(n, 2))
    instance_id = f"random_n{n}_seed{seed or 0}"
    return instance_id, coords


def prepare_test_inputs(
    tsplib_paths: list[str | pathlib.Path] | None = None,
    random_specs: list[tuple[int, int | None]] | None = None,
) -> list[TSPInstance]:
    """Prepare test inputs for FunSearch evaluator.

    Returns TSPInstance objects (hashable, sortable by instance_id) for use
    as ProgramsDatabase test input keys.

    Args:
        tsplib_paths: Paths to TSPLib files.
        random_specs: List of (n_cities, seed) for random instances.

    Returns:
        List of TSPInstance for test_inputs.
    """
    instances: list[TSPInstance] = []

    if tsplib_paths:
        for p in tsplib_paths:
            try:
                name, coords = load_tsplib(p)
                dist_matrix = coords_to_dist_matrix(coords)
                instances.append(TSPInstance(name, dist_matrix))
            except Exception as e:
                print(f"Warning: Could not load {p}: {e}")

    if random_specs:
        for n, seed in random_specs:
            inst_id, coords = generate_random_euclidean_tsp(n, seed)
            dist_matrix = coords_to_dist_matrix(coords)
            instances.append(TSPInstance(inst_id, dist_matrix))

    if not instances:
        # Fallback: use small random instances
        for n, s in [(10, 42), (15, 123), (20, 456)]:
            inst_id, coords = generate_random_euclidean_tsp(n, s)
            dist_matrix = coords_to_dist_matrix(coords)
            instances.append(TSPInstance(inst_id, dist_matrix))

    return instances
