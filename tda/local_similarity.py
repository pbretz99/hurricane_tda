from gudhi.wasserstein import wasserstein_distance

import xarray as xr
import numpy as np

from handling_data.subsetting_bbox import SubsetBBox
from tda.persistence import gudhi_cubical_persistence_wrapper


def run_local_comparison(
        da: xr.DataArray,
        lat: float, 
        lon: float, 
        r: float = 1.0,
        skip: int = 25,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
    bbox = SubsetBBox.from_center_and_radius(lat, lon, r)
    da_small = bbox.subset_xr(da)
    pers_base = gudhi_cubical_persistence_wrapper(da_small.values)

    lat_with_skip = da["lat"].values[::skip]
    lon_with_skip = da["lon"].values[::skip]
    dist_mat_0 = np.zeros((len(lat_with_skip), len(lon_with_skip)))
    dist_mat_1 = np.zeros(dist_mat_0.shape)
    for i, lat_c in enumerate(lat_with_skip):
        for j, lon_c in enumerate(lon_with_skip):
            if verbose: print(f"Running ({lat_c:.2f}, {lon_c:.2f})")
            bbox = SubsetBBox.from_center_and_radius(lat_c, lon_c, r)
            da_small_current = bbox.subset_xr(da)
            pers_current = gudhi_cubical_persistence_wrapper(da_small_current.values)
            dist_mat_0[i,j] = wasserstein_distance(pers_base[0], pers_current[0], order=2)
            dist_mat_1[i,j] = wasserstein_distance(pers_base[1], pers_current[1], order=2)
    
    return dist_mat_0, dist_mat_1
