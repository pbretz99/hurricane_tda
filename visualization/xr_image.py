import matplotlib.pyplot as plt
import xarray as xr

from handling_data.xr_metadata import XRMetadata


def base_plot_da(ax: plt.Axes, da: xr.DataArray, **kwargs) -> plt.AxesImage:
    extent = XRMetadata.from_xr(da)
    im = ax.imshow(da.values, origin="lower", extent=extent, **kwargs)
    ax.set_xlabel("Lon. [deg]")
    ax.set_ylabel("Lat. [deg]")
    return im
