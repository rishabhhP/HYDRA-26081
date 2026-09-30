"""Read dataset metadata without loading complete weather fields."""
import json
import sys
from pathlib import Path

import xarray as xr


def inspect(path):
    with xr.open_dataset(path) as ds:
        return {
            "file": str(path),
            "dimensions": dict(ds.sizes),
            "attributes": {k: str(v) for k, v in ds.attrs.items()},
            "coordinates": {
                k: {"first": str(v.values.flat[0]), "last": str(v.values.flat[-1]),
                    "units": v.attrs.get("units"), "size": v.size}
                for k, v in ds.coords.items() if v.size
            },
            "variables": {
                k: {"dimensions": list(v.dims), "units": v.attrs.get("units"),
                    "long_name": v.attrs.get("long_name"),
                    "fill_value": str(v.encoding.get("_FillValue"))}
                for k, v in ds.data_vars.items()
            },
        }


if __name__ == "__main__":
    root = Path(sys.argv[1])
    print(json.dumps([inspect(p) for p in sorted(root.rglob("*.nc"))], indent=2))
