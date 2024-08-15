import numpy as np

from handling_data.downloading_goes import GOESRequestInfo, GOESProduct
from handling_data.subsetting_bbox import SubsetBBox

from tda.local_similarity import run_local_comparison

from visualization.plotting import add_square, add_circ
from visualization.xr_image import base_plot_da, base_plot_vals, get_extent

import matplotlib.pyplot as plt


if __name__ == "__main__":
    request_info = GOESRequestInfo(2017, 8, 26, 0, GOESProduct.GOES15)
    subset_bbox = SubsetBBox.from_center_and_radius(27.9, -96.6, 10)
    lat_c, lon_c = (26, -99)
    r = 2.0
    
    #request_info.download()
    ds = request_info.load()
    da = subset_bbox.subset_xr(ds["ch4"].isel(time=0))

    dist_0, dist_1 = run_local_comparison(da, lat_c, lon_c, r, skip=10, circle_subset=True)
    #dist_1 = np.clip(dist_1, 0, 25)
    
    fig, ax = plt.subplots()
    base_plot_da(ax, da)
    add_circ(ax, lon_c, lat_c, r, c="red", ls="--")
    
    extent = get_extent(da)
    fig, ax = plt.subplots()
    base_plot_vals(ax, dist_0, extent)
    add_circ(ax, lon_c, lat_c, r, c="red", ls="--")
    ax.set_title("W2B0")
    
    fig, ax = plt.subplots()
    base_plot_vals(ax, dist_1, extent)
    add_circ(ax, lon_c, lat_c, r, c="red", ls="--")
    ax.set_title("W2B1")

    plt.show()
