from handling_data.downloading_goes import GOESRequestInfo, GOESProduct
from handling_data.subsetting_bbox import Bounds, SubsetBBox
from handling_data.xr_metadata import XRMetadata

import matplotlib.pyplot as plt


if __name__ == "__main__":
    request_info = GOESRequestInfo(2017, 8, 26, 0, GOESProduct.GOES15)
    subset_bbox = SubsetBBox(
        lat=Bounds(15, 35),
        lon=Bounds(-105, -85),
    )
    
    #request_info.download()
    ds = request_info.load()
    da = subset_bbox.subset_xr(ds["ch4"].isel(time=0))
    vals = da.values

    extent = XRMetadata.from_xr(da).extent()

    fig, ax = plt.subplots()
    ax.imshow(vals, origin="lower", extent=extent)
    plt.show()
