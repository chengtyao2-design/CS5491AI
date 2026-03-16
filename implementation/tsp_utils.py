"""TSP utilities: TSPLib parsing, distance matrix, and instance generation."""
import logging
import pathlib
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tsplib95
    _TSPLIB95_AVAILABLE = True
except ImportError:
    _TSPLIB95_AVAILABLE = False


class TSPInstance:
    """Hashable TSP instance for use as ProgramsDatabase test input key."""

    def __init__(
        self,
        instance_id: str,
        dist_matrix: np.ndarray,
        optimal_tour_length: float | None = None,
    ):
        self.instance_id = instance_id
        self.dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
        self.optimal_tour_length = optimal_tour_length

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


def _load_tsplib_legacy(
    filepath: str | pathlib.Path,
) -> tuple[str, np.ndarray, float | None]:
    """Legacy parser for NODE_COORD_SECTION + EUC_2D. Returns (name, coords, optimal)."""
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"TSPLib file not found: {filepath}")

    name = path.stem
    coords: list[list[float]] = []
    optimal: float | None = None
    in_section = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not in_section:
                for key in ("OPTIMUM", "BEST_KNOWN", "OPTIMAL_VALUE"):
                    if line.upper().startswith(key) and ":" in line:
                        try:
                            optimal = float(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                        break
            if line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if in_section:
                if line in ("EOF", ""):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])

    coords_arr = np.array(coords, dtype=np.float64)
    return name, coords_arr, optimal


def load_opt_tour(filepath: str | pathlib.Path) -> list[int]:
    """Parse TSPLib .opt.tour file and return tour as list of 1-based node IDs.

    Args:
        filepath: Path to .opt.tour file.

    Returns:
        List of node IDs in tour order (1-based, TSPLib convention).
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Opt tour file not found: {filepath}")

    tour: list[int] = []
    in_section = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_section = True
                continue
            if in_section:
                if line in ("EOF", "-1", ""):
                    break
                try:
                    tour.append(int(line))
                except ValueError:
                    pass

    return tour


def compute_tour_length(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Compute total length of a closed tour given distance matrix.

    TSPLib uses 1-based node IDs; dist_matrix uses 0-based indices.

    Args:
        tour: List of 1-based node IDs in tour order.
        dist_matrix: (n, n) distance matrix, 0-indexed.

    Returns:
        Total tour length (closed loop).
    """
    if len(tour) < 2:
        return 0.0
    n = len(dist_matrix)
    length = 0.0
    for i in range(len(tour) - 1):
        a, b = tour[i] - 1, tour[i + 1] - 1
        if 0 <= a < n and 0 <= b < n:
            length += dist_matrix[a, b]
    # Close the loop
    a, b = tour[-1] - 1, tour[0] - 1
    if 0 <= a < n and 0 <= b < n:
        length += dist_matrix[a, b]
    return length


def load_tsplib(
    filepath: str | pathlib.Path,
    opt_tour_path: str | pathlib.Path | None = None,
) -> tuple[str, np.ndarray, float | None]:
    """Parse TSPLib file and return (name, dist_matrix, optimal).

    Supports EUC_2D, ATT, GEO, EXPLICIT via tsplib95 when available.
    Falls back to legacy NODE_COORD_SECTION + EUC_2D parser otherwise.
    When .tsp has no OPTIMUM/BEST_KNOWN, automatically tries <stem>.opt.tour
    in the same directory (or opt_tour_path if specified).

    Args:
        filepath: Path to .tsp file.
        opt_tour_path: Optional path to .opt.tour; if None, auto-lookup
            path.parent / (path.stem + ".opt.tour").

    Returns:
        Tuple of (instance_name, dist_matrix, optimal_tour_length) where
        optimal_tour_length is from OPTIMUM/BEST_KNOWN, .opt.tour, or None.
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"TSPLib file not found: {filepath}")

    optimal_source: str | None = None  # ".tsp", ".opt.tour", or None

    if _TSPLIB95_AVAILABLE:
        try:
            problem = tsplib95.load(path)
            n = problem.dimension
            dist_matrix = np.zeros((n, n), dtype=np.float64)
            nodes = list(problem.get_nodes())
            for i, ni in enumerate(nodes):
                for j, nj in enumerate(nodes):
                    dist_matrix[i, j] = problem.get_weight(ni, nj)
            optimal = getattr(problem, "best_known", None) or getattr(
                problem, "optimum", None
            )
            if optimal is not None:
                optimal = float(optimal)
                optimal_source = ".tsp"
            name = problem.name or path.stem
        except Exception:
            name, coords, optimal = _load_tsplib_legacy(path)
            dist_matrix = coords_to_dist_matrix(coords)
            if optimal is not None:
                optimal_source = ".tsp"
    else:
        name, coords, optimal = _load_tsplib_legacy(path)
        dist_matrix = coords_to_dist_matrix(coords)
        if optimal is not None:
            optimal_source = ".tsp"

    # Fallback to .opt.tour when optimal is missing
    if optimal is None:
        opt_path = pathlib.Path(opt_tour_path) if opt_tour_path else path.parent / (
            path.stem + ".opt.tour"
        )
        if opt_path.exists():
            try:
                tour = load_opt_tour(opt_path)
                if len(tour) == dist_matrix.shape[0]:
                    optimal = compute_tour_length(tour, dist_matrix)
                    optimal_source = ".opt.tour"
            except Exception as e:
                logger.debug("Failed to load %s: %s", opt_path, e)

    # Log optimal source for user visibility
    if optimal is not None and optimal_source == ".tsp":
        logger.info(
            "Instance %s: optimal=%.2f (from .tsp OPTIMUM/BEST_KNOWN)", name, optimal
        )
    elif optimal is not None and optimal_source == ".opt.tour":
        logger.info("Instance %s: optimal=%.2f (from .opt.tour)", name, optimal)
    elif optimal is None:
        logger.warning(
            "Instance %s: no optimal value; gap chart will be disabled. "
            "Add OPTIMUM to .tsp or provide .opt.tour",
            name,
        )

    return name, dist_matrix, optimal


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
                name, dist_matrix, optimal = load_tsplib(p)
                instances.append(TSPInstance(name, dist_matrix, optimal))
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

    # Summary: warn if any TSPLib instances lack optimal (gap disabled)
    no_opt = [
        i.instance_id
        for i in instances
        if not i.instance_id.startswith("random_")
        and getattr(i, "optimal_tour_length", None) is None
    ]
    if no_opt:
        logger.warning(
            "Instances without optimal (gap chart disabled): %s", no_opt
        )

    return instances
