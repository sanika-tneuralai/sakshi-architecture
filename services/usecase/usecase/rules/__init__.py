"""
Usecase/rules/__init__.py - Auto-discovering rules registry.

How auto-discovery works:
    1. At startup, scan usecase/rules/ AND all domain subdirectories for .py files
    2. Import each module
    3. Find all classes that:
        - Subclass BaseUsecaseRule
        - Are not BaseUsecaseRule itself
        - Have a USECASE_ID class attribute (the registration key)
    4. Register them automatically

Domain subdirectory structure (add new domains freely — no code changes needed here):
    usecase/rules/
    ├── stores/       ← retail/store usecases
    ├── vehicles/     ← parking, compliance, extraction
    ├── saftey/       ← fire, smoke, gun
    └── hotel/        ← future domain (just create the folder + files)

How to register a new usecase:
    1. Create usecase/rules/<domain>/my_new_usecase.py
    2. Subclass BaseUsecaseRule
    3. Add: USECASE_ID = "my_new_usecase"  (must be globally unique)
    4. Implement evaluate()
    Done. Nothing else to change.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Type

from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

# Registry: maps usecase_id string -> rule class
# Populated automatically at module import time
USECASE_REGISTRY: Dict[str, Type[BaseUsecaseRule]] = {}


def _register_from_module(module_name: str) -> None:
    """Import one module and register any BaseUsecaseRule subclasses found in it."""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        logger.error("[REGISTRY] Failed to import module %s: %s", module_name, e)
        return

    for name, cls in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(cls, BaseUsecaseRule)
            and cls is not BaseUsecaseRule
            and hasattr(cls, "USECASE_ID")
        ):
            usecase_id = cls.USECASE_ID
            if usecase_id in USECASE_REGISTRY:
                logger.warning(
                    "[REGISTRY] Duplicate USECASE_ID '%s' in %s.%s — skipping. "
                    "Already registered by: %s",
                    usecase_id, module_name, name, USECASE_REGISTRY[usecase_id],
                )
                continue
            USECASE_REGISTRY[usecase_id] = cls
            logger.info("[REGISTRY] Registered: %s -> %s", usecase_id, cls.__name__)


def _discover_rules() -> None:
    """
    Recursively scan rules/ and all domain subdirectories.

    Flat files directly in rules/ are still supported so existing rules
    don't need to be moved. Domain subfolders (stores/, vehicles/, etc.)
    are discovered automatically — just create a new folder with an
    __init__.py and drop rule files inside it.
    """
    rules_dir = Path(__file__).parent

    # --- Pass 1: flat files directly in rules/ (backward compatible) ---
    for module_info in pkgutil.iter_modules([str(rules_dir)]):
        if module_info.name in ("base",):
            continue
        if module_info.ispkg:
            # handled in pass 2
            continue
        _register_from_module(f"usecase.rules.{module_info.name}")

    # --- Pass 2: domain subdirectories ---
    for subdir in sorted(rules_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("_"):
            continue  # skip __pycache__ etc.

        domain = subdir.name  # e.g. "vehicles", "stores", "saftey"
        init_file = subdir / "__init__.py"
        if not init_file.exists():
            logger.warning(
                "[REGISTRY] Skipping domain folder '%s' — missing __init__.py", domain
            )
            continue

        for module_info in pkgutil.iter_modules([str(subdir)]):
            module_name = f"usecase.rules.{domain}.{module_info.name}"
            _register_from_module(module_name)


def get_usecase_rule(usecase_id: str) -> BaseUsecaseRule:
    """
    Get a rule instance by usecase_id.

    Args:
        usecase_id: e.g. "parking_detection" or "bag_detection"

    Returns:
        Rule instance ready to call .evaluate()

    Raises:
        ValueError: if usecase_id is not in the registry
    """
    if usecase_id not in USECASE_REGISTRY:
        available = ", ".join(sorted(USECASE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown usecase: '{usecase_id}'. Available: [{available}]"
        )
    rule_class = USECASE_REGISTRY[usecase_id]
    return rule_class(usecase_id)


def list_usecases() -> list:
    """Return all registered usecase IDs, sorted. Useful for validation and docs."""
    return sorted(USECASE_REGISTRY.keys())


# Keep backward-compatible alias
list_usecase = list_usecases


_discover_rules()
logger.info(
    "[REGISTRY] Discovery complete. %d usecases registered: %s",
    len(USECASE_REGISTRY), list_usecases(),
)




