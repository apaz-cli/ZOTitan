"""
Importing this imports every file in the folder for its `@register_objective`
side effect, so the names land in `objective.OBJECTIVES`.
"""
import importlib
import pkgutil

for _info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_info.name}")
