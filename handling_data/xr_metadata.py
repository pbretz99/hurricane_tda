from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr

from handling_data.bounds import Bounds


@dataclass
class XRMetadata:
    lat_bounds: Bounds
    lon_bounds: Bounds
    dlat: float
    dlon: float

    @classmethod
    def from_xr(cls, ds: xr.Dataset | xr.DataArray) -> Self:
        lat_grid = ds["lat"].values
        lon_grid = ds["lon"].values
        return cls(
            lat_bounds=_bounds_of_grid(lat_grid),
            lon_bounds=_bounds_of_grid(lon_grid),
            dlat=_grid_spacing(lat_grid),
            dlon=_grid_spacing(lon_grid),
        )
    
    def extent(self) -> list[float]:
        return [
            self.lon_bounds.lower - self.dlon / 2,
            self.lon_bounds.upper + self.dlon / 2,
            self.lat_bounds.lower - self.dlat / 2,
            self.lat_bounds.upper + self.dlat / 2,
        ]


def _bounds_of_grid(grid: np.ndarray) -> Bounds:
    assert np.all(np.diff(grid) > 0)
    return Bounds(grid[0], grid[-1])

def _grid_spacing(grid: np.ndarray, tol: float = 0.0001) -> float:
    diffs = np.diff(grid)
    assert np.all(diffs > 0)
    deviation = np.abs(diffs - diffs[0])
    assert np.all(deviation < tol)
    return diffs[0]
