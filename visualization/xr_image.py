import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from handling_data.xr_metadata import XRMetadata


def get_extent(da: xr.DataArray) -> list[float]:
    return XRMetadata.from_xr(da).extent()


def base_plot_da(ax: plt.Axes, da: xr.DataArray, **kwargs):
    return base_plot_vals(ax, da.values, extent=get_extent(da), **kwargs)


def base_plot_vals(ax: plt.Axes, vals: np.ndarray, extent: list[float], **kwargs):
    im = ax.imshow(vals, origin="lower", extent=extent, **kwargs)
    ax.set_xlabel("Lon. [deg]")
    ax.set_ylabel("Lat. [deg]")
    return im
