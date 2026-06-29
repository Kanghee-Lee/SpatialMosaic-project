from pathlib import Path

# Expose the actual VGGT source tree under this top-level package so imports
# like `vggt.models...` work when running from the monorepo checkout.
_inner_pkg = Path(__file__).resolve().parent / "vggt"
if _inner_pkg.is_dir():
    __path__.append(str(_inner_pkg))
