if __package__:
    from .region_edit_toolkit.registry import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
else:
    # Pytest imports this file as the top-level module ``__init__`` when the
    # checkout directory contains hyphens.  Load one synthetic package so the
    # same relative-import path is exercised instead of maintaining a second
    # test-only registry.
    import importlib.util
    from pathlib import Path
    import sys

    _package_name = "_comfyui_region_edit_toolkit_static"
    _package_root = Path(__file__).resolve().parent
    _package = sys.modules.get(_package_name)
    if _package is None:
        _spec = importlib.util.spec_from_file_location(
            _package_name,
            _package_root / "__init__.py",
            submodule_search_locations=[str(_package_root)],
        )
        if _spec is None or _spec.loader is None:
            raise ImportError("unable to create the static package import specification")
        _package = importlib.util.module_from_spec(_spec)
        sys.modules[_package_name] = _package
        _spec.loader.exec_module(_package)
    NODE_CLASS_MAPPINGS = _package.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _package.NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
