from dataclasses import dataclass

import xarray as xr

from handling_data.bounds import Bounds


@dataclass(frozen=True)
class SubsetBBox:
    lat: Bounds
    lon: Bounds

    def subset_xr(self, da: xr.DataArray) -> xr.DataArray:
        assert "lat" in da.dims
        assert "lon" in da.dims
        da = da.where(da["lat"] >= self.lat.lower, drop=True)
        da = da.where(da["lat"] <= self.lat.upper, drop=True)
        da = da.where(da["lon"] >= self.lon.lower, drop=True)
        da = da.where(da["lon"] <= self.lon.upper, drop=True)
        return da
