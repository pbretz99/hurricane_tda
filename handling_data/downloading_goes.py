from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import requests
import xarray as xr

from definitions import BASE_GOES_URL, DEFAULT_SAVE_DIR
from exceptions import DownloadException, LoadingException


class GOESProduct(StrEnum):
    GOES13 = "goes13"
    GOES15 = "goes15"


VALID_PRODUCTS: dict[tuple[int, int], list[GOESProduct]] = {
    (2017, 1): [GOESProduct.GOES15],
    (2017, 2): [GOESProduct.GOES15],
    (2017, 3): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 4): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 5): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 6): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 7): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 8): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 9): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 10): [GOESProduct.GOES13, GOESProduct.GOES15],
    (2017, 11): [GOESProduct.GOES13],
    (2017, 12): [GOESProduct.GOES13],
}


@dataclass
class GOESRequestInfo:
    year: int
    month: int
    day: int
    hour: int
    product: GOESProduct

    def validate(self) -> None:
        year_month_pair = (self.year, self.month)
        if year_month_pair not in VALID_PRODUCTS.keys():
            raise DownloadException(f"Year-month pair {year_month_pair} not in keys of valid products")
        if self.product not in VALID_PRODUCTS[year_month_pair]:
            raise DownloadException(f"Product {self.product} not in valid products for year-month {year_month_pair}: {VALID_PRODUCTS[year_month_pair]}")
        try:
            datetime(self.year, self.month, self.day, self.hour)
        except ValueError:
            raise DownloadException(f"Invalid datetime for request {self}")

    def filename(self) -> str:
        return f"GridSat-GOES.{self.product.value}.{self.year}.{self.month:02d}.{self.day:02d}.{self.hour:02d}00.v01.nc"

    def request_url(self) -> str:
        base_url = f"{BASE_GOES_URL}/{self.year}/{self.month:02d}"
        return f"{base_url}/{self.filename()}"

    def download(self, save_dir: Path = DEFAULT_SAVE_DIR) -> None:
        url = self.request_url()
        savefile = save_dir.joinpath(self.filename())
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(savefile, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"GOES download complete for {self}")

    def load(self, save_dir: Path = DEFAULT_SAVE_DIR) -> xr.Dataset:
        savefile = save_dir.joinpath(self.filename())
        if not savefile.exists():
            raise LoadingException(f"Savefile does not exist with path {savefile}")
        return xr.open_dataset(savefile)
