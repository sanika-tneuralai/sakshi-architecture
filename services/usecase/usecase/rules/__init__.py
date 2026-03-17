"""
Usecase/rules/__init__.py - Auto-discovering rules registry.

How auto discover works:
    1. At startup, scan the usecase/rules/directory for .py files
    2. import each module
    3. find all classes that:
        -subclass BaseUsecaseRule
        -Are not BaseUsecaseRule itself
        -Have a USECASE_ID class attribute (the registration Key)
    4. Register them automatically

Result: Adding a new usecase = create the file. Nothing else.

How to register a new usecase:
       1. Create usecase/rules/my_new_usecase.py
       2. Subclass BaseUsecaseRule, 
       3. Add: USECASE_ID = "my_new_usecase" (must be unique across all usecases, this is the registry key)
       4.Implement evaluate()
       Done. No __init__.py changes needed
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Type

from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

# The reistry: maps usecases_id string -> rule class
# Populated automatically at module import time
USECASE_REGISTRY: Dict[str, Type[BaseUsecaseRule]] = {}

def _discover_rules() -> None:
    """Scan this package directory and register all BaseUsecaseRule subclasses
    
    When the service starts, python imports usecase/rules/__init__.py 
    Discovery runs immediately, so by the time the first request arrives, the registry is fully populated. No lazy loading, no race conditions.
    
    """
    rules_dir = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(rules_dir)]):
        if module_info.name in ("base", "__init__"):
            continue # Skip base and __init__.py

        module_name = f"usecase.rules.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"[REGISTRY]Failed to import module {module_name}: {e}")
            continue

        for name,cls in inspect.getmembers(module,inspect.isclass): #find all BaseUsecaseRule subclasses in this module
          if(
              issubclass(cls, BaseUsecaseRule)
              and cls is not BaseUsecaseRule
              and hasattr(cls, "USECASE_ID") #must declare it's own ID
          ):
              usecase_id = cls.USECASE_ID
              if usecase_id in USECASE_REGISTRY:
                  logger.warning(
                      f"[REGISTRY]Duplicate USECASE_ID '{usecase_id}' found in {module_name}.{name}. Skipping registration. Previous class: {USECASE_REGISTRY[usecase_id]}"
                  )
                  continue

              USECASE_REGISTRY[usecase_id] = cls
              logger.info(f'[REGISTRY] Registered usecase rule: {usecase_id} -> {cls.__name__}')

def get_usecase_rule(usecase_id: str) -> BaseUsecaseRule:
    """ Get a rule instance by usecase_id.
    Unchanged interface from your current code - callers don't need to change.

    Args:
    usecase_id: e.g. "person_in_roi"

    Returns:
    Rule instance ready to call .evaluate()

    Raises:
    ValueError:  if usecase_id not in registry
    """
    if usecase_id not in USECASE_REGISTRY:
        available = ", ".join((sorted(USECASE_REGISTRY.keys())))
        raise ValueError(f'unknown usecase: {usecase_id}, Available;[{available}]')
    rule_class = USECASE_REGISTRY[usecase_id]
    return rule_class(usecase_id)
    
def list_usecase() -> list:
    """
    Return all registered usecase IDs. Useful for validation and docs.
    """
    return sorted(USECASE_REGISTRY.keys())

_discover_rules()
logger.info(f'[REGISTRY] Discovery complete. {len(USECASE_REGISTRY)} usecase registered: {list_usecase()}')




