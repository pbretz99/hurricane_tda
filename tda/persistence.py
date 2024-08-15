from gudhi import CubicalComplex

import numpy as np


def gudhi_cubical_persistence_wrapper(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cc = CubicalComplex(top_dimensional_cells=vals)
    cc.compute_persistence(min_persistence=0.01)
    persistence_0: np.ndarray = cc.persistence_intervals_in_dimension(0)
    persistence_1: np.ndarray = cc.persistence_intervals_in_dimension(1)
    return persistence_0, persistence_1
