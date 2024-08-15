from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float

    def __post_init__(self):
        assert self.lower <= self.upper
