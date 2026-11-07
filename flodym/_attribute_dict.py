"""Home to a dictionary whose values can also be accessed as attributes."""

import keyword
import logging
import re
from typing import Any, TypeVar, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

logger = logging.getLogger(__name__)

ValueT = TypeVar("ValueT")

_ARROW_PATTERN = re.compile(r"\s*=>\s*")
_NON_WORD_PATTERN = re.compile(r"\W+")


def _to_attribute_name(key: str) -> str:
    """Convert a dictionary key to a valid Python identifier.

    Arrows (``=>``) including surrounding whitespace are replaced by ``__to__``, every other run of
    characters not allowed in identifiers is replaced by a single ``_``. A leading digit is prefixed
    with ``_``, and Python keywords get a trailing ``_``.

    Args:
        key: The dictionary key.

    Returns:
        The attribute name under which the value is accessible.

    Example:
        >>> _to_attribute_name("sysenv => process 1")
        'sysenv__to__process_1'
    """
    name = _ARROW_PATTERN.sub("__to__", key)
    name = _NON_WORD_PATTERN.sub("_", name)
    if not name or name[0].isdigit():
        name = "_" + name
    if keyword.iskeyword(name):
        name = name + "_"
    return name


class AttributeDict(dict[str, ValueT]):
    """A dictionary whose values can also be accessed as attributes.

    ``flows["foo"]`` is also accessible as ``flows.foo``. Keys that are not valid Python identifiers
    are accessible under a sanitized alias (see :py:func:`_to_attribute_name`), e.g.
    ``flows["sysenv => process 1"]`` as ``flows.sysenv__to__process_1``.
    Use ``dir()`` or tab completion to see all available aliases.

    Keys are excluded from attribute access, with a warning, if

    - their alias coincides with the alias of another key, or
    - their alias coincides with a dict attribute or method, such as ``values`` or ``items``.

    Such keys are still accessible via ``[]``.

    Attribute assignment is not supported. To change the values of an array in place, use
    ``flows.foo[...] = new_values``; to replace a dictionary entry, use ``flows["foo"] = new_value``.
    """

    def _alias_map(self) -> dict[str, str]:
        """Return the mapping from attribute aliases to keys, rebuilding it if the keys changed."""
        keys = frozenset(self.keys())
        cache = self.__dict__.get("_alias_cache")
        if cache is not None and cache[0] == keys:
            return cache[1]
        alias_map = self._build_alias_map()
        object.__setattr__(self, "_alias_cache", (keys, alias_map))
        return alias_map

    def _build_alias_map(self) -> dict[str, str]:
        """Map aliases to keys, excluding ambiguous aliases and aliases of dict attributes."""
        keys_by_alias: dict[str, list[str]] = {}
        for key in self.keys():
            if isinstance(key, str):
                keys_by_alias.setdefault(_to_attribute_name(key), []).append(key)

        alias_map: dict[str, str] = {}
        for alias, keys in keys_by_alias.items():
            if len(keys) > 1:
                logger.warning(
                    f"Keys {keys} all map to the attribute name '{alias}'. "
                    "They are excluded from attribute access; use [] instead."
                )
            elif hasattr(type(self), alias):
                logger.warning(
                    f"The attribute name '{alias}' of key '{keys[0]}' is a dict attribute. "
                    "The key is excluded from attribute access; use [] instead."
                )
            else:
                alias_map[alias] = keys[0]
        return alias_map

    def __getattr__(self, name: str) -> ValueT:
        """Return the value whose key has the attribute alias ``name``."""
        if name.startswith("__") or name == "_alias_cache":
            raise AttributeError(name)
        alias_map = self._alias_map()
        if name in alias_map:
            return self[alias_map[name]]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Available attribute names: {sorted(alias_map)}. Keys can also be accessed via []."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Attribute assignment is not supported for {type(self).__name__}. "
            f"To change the values of an array in place, use .{name}[...] = new_values; "
            f"to replace an entry, use ['<key>'] = new_value."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"Attribute deletion is not supported for {type(self).__name__}. Use del [<key>]."
        )

    def __dir__(self) -> list[str]:
        """List the dict attributes together with the attribute aliases of the keys."""
        return sorted(set(super().__dir__()) | set(self._alias_map()))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Validate as ``dict[str, ValueT]``, then convert to an `AttributeDict`."""
        args = get_args(source_type)
        values_schema = handler.generate_schema(args[0]) if args else core_schema.any_schema()
        dict_schema = core_schema.dict_schema(
            keys_schema=core_schema.str_schema(), values_schema=values_schema
        )
        return core_schema.no_info_after_validator_function(cls, dict_schema)
