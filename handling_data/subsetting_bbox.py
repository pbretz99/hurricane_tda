from dataclasses import dataclass
from typing import Self

import xarray as xr

from handling_data.bounds import Bounds


@dataclass(frozen=True)
class SubsetBBox:
    lat: Bounds
    lon: Bounds

    @classmethod
    def from_center_and_radius(cls, lat_center: float, lon_center: float, radius: float) -> Self:
        return cls(
            lat=Bounds(lat_center-radius, lat_center+radius),
            lon=Bounds(lon_center-radius, lon_center+radius),
        )

    def subset_xr(self, da: xr.DataArray) -> xr.DataArray:
        assert "lat" in da.dims
        assert "lon" in da.dims
        da = da.where(da["lat"] >= self.lat.lower, drop=True)
        da = da.where(da["lat"] <= self.lat.upper, drop=True)
        da = da.where(da["lon"] >= self.lon.lower, drop=True)
        da = da.where(da["lon"] <= self.lon.upper, drop=True)
        return da
