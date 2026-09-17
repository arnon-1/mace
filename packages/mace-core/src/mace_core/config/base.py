"""The base class every v1 configuration schema derives from.

A configuration is a Pydantic model tree: the top level subclasses
`ReforgeBaseConfig`, each nested section subclasses `ConfigSection`. Values come
from three layers, lowest precedence first:

    schema defaults < one config file (.toml/.yaml/.yml/.json) < dotted CLI overrides

Nothing else feeds a config: no environment variables, no dotenv files, so a
run is reproducible from its file and its command line alone.

Unknown keys are hard errors. Pydantic detects them (`extra="forbid"` on every
level of the tree); this module turns each into a message that names the key
by its dotted path and, when there is one, the nearest valid neighbour.

Field types are restricted to what survives a JSON round trip unchanged, so
that the resolved export is a fixed point: `set` and `frozenset` fields are
rejected when the schema class is defined, because their element order is not
stable across interpreter runs. Use a list.
"""

from __future__ import annotations

import difflib
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import CliSettingsSource, SettingsError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["ConfigError", "ConfigSection", "ReforgeBaseConfig", "read_config_file"]

#: Config file extensions this module reads, keyed to their parsers.
_FILE_PARSERS = {
    ".toml": tomllib.loads,
    ".yaml": yaml.safe_load,
    ".yml": yaml.safe_load,
    ".json": json.loads,
}


def _sections_in(
    annotation: Any, inside: bool = False
) -> Iterator[tuple[type[BaseModel], bool]]:
    """Every section class reachable from a field annotation, with whether it
    sits inside a dict/list/tuple (so an error location has a key or index
    before its fields)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation, inside
        return
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for arg in get_args(annotation):
            yield from _sections_in(arg, inside)
    elif origin in (dict, list, tuple):
        for arg in get_args(annotation):
            yield from _sections_in(arg, True)


def _section_of(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """The one section a field can hold, and whether it is inside a collection."""
    found = dict(_sections_in(annotation))
    return next(iter(found.items())) if found else (None, False)


def _holds_set(annotation: Any) -> bool:
    origin = get_origin(annotation) or annotation
    return origin in (set, frozenset) or any(
        _holds_set(a) for a in get_args(annotation)
    )


def _check_schema(model: type[BaseModel]) -> None:
    """Fail at class definition for a field shape the contract cannot keep.

    Each rule protects one guarantee: no sets (order is not stable across
    runs, so the export would not be a fixed point); no aliases or computed
    fields (the export would not validate back); one section class per field
    (a union of sections has no single set of valid keys to suggest); every
    section rejects unknown keys.
    """

    def reject(name: str, reason: str) -> None:
        raise TypeError(f"{model.__name__}.{name} {reason}")

    for name in model.model_computed_fields:
        reject(name, "is a computed field; the export must validate back, so drop it")
    for name, field in model.model_fields.items():
        if field.alias or field.validation_alias or field.serialization_alias:
            reject(name, "has an alias; config keys are field names, so drop it")
        if _holds_set(field.annotation):
            reject(
                name,
                "is typed as a set; set order is not stable across runs. Use a list",
            )
        sections = {section for section, _ in _sections_in(field.annotation)}
        if len(sections) > 1:
            reject(name, "is a union of sections; use one section with a `kind` field")
        for section in sections:
            if section.model_config.get("extra") != "forbid":
                reject(
                    name,
                    f"holds {section.__name__}, which accepts unknown keys; "
                    f"subclass ConfigSection",
                )


class ConfigError(ValueError):
    """A config file or override the schema rejects.

    Raised for a missing, unparsable or malformed file, an unknown key and an
    override the CLI parser cannot make sense of. The message names the file
    or the offending key by its dotted path, and suggests the nearest valid
    key when there is a close match.
    """


class ConfigSection(BaseModel):
    """A nested section of a configuration. Unknown keys are errors here too."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        _check_schema(cls)


class ReforgeBaseConfig(ConfigSection):
    """Root of a configuration tree. Subclass it; nest `ConfigSection`s in it.

    `load()` is the one entry point that reads a file and applies overrides.
    Constructing the class directly behaves like a plain Pydantic model, which
    keeps tests and programmatic construction free of any file or CLI plumbing.

    Deliberately a plain pydantic model, not `pydantic_settings.BaseSettings`:
    that class merges environment variables and dotenv files in front of
    validation, case-insensitively and with private constructor options
    (`_env_file`, ...) that a config file could set. The root is validated
    exactly like every section below it. pydantic-settings is used for one
    thing only, turning the argv list into a nested dict.
    """

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        cli_overrides: Sequence[str] = (),
    ) -> Self:
        """Build the config from defaults, then the file, then the overrides.

        `cli_overrides` is the argument list after the program name, e.g.
        `["--model.num_interactions", "3", "--seed=7"]`; the dotted path names
        a field at any depth of the tree. Values are parsed against the
        field's type by pydantic-settings, so lists and `null` work as well as
        scalars. A value that itself starts with `--` has to be written
        `--name=--value`. A whole section, and a dict-valued field, take a JSON
        value (`--model '{"num_interactions": 3}'`); entries of a dict cannot
        be addressed by dotted path.

        Raises `ConfigError` for an unknown key, an unreadable file or an
        unparsable override, and pydantic's `ValidationError` for a value of
        the wrong type. Direct construction skips this and raises pydantic's
        `ValidationError` for an unknown key too, without a neighbour.
        """
        values: dict[str, Any] = {}
        if config_file is not None:
            values = read_config_file(config_file)
        if cli_overrides:
            _check_override_names(cls, cli_overrides)
            try:
                cli_values = CliSettingsSource(
                    cls,  # ty: ignore[invalid-argument-type]  # any BaseModel works
                    cli_parse_args=list(cli_overrides),
                    cli_exit_on_error=False,
                )()
            except SettingsError as error:
                raise ConfigError(f"cannot parse CLI overrides: {error}") from error
            values = _deep_update(values, cli_values)
        try:
            return cls.model_validate(values)
        except ValidationError as error:
            unknown = _unknown_key_messages(cls, error)
            if not unknown:
                raise
            raise ConfigError("\n".join(unknown)) from error

    def to_resolved_dict(self) -> dict[str, Any]:
        """Every field, defaults included, as JSON-native values.

        Keys follow schema declaration order at each level, whatever order the
        input had; a dict-valued field keeps the order it was given. Writing
        the result to any of the three file formats and loading it back gives
        an identical config, and resolving that gives an identical dict: the
        export is a fixed point. (TOML has no null, so a
        config holding a `None` can only go back out as YAML or JSON.)
        """
        return self.model_dump(mode="json")

    def to_user_dict(self) -> dict[str, Any]:
        """Only the fields the file and the overrides set, as JSON-native values.

        The complement of `to_resolved_dict()`: what the user actually wrote,
        with defaults left out, for the model metadata's record of user-set
        configuration.
        """
        return self.model_dump(mode="json", exclude_unset=True)


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Parse one config file, choosing the parser by extension.

    An empty file is an empty config. Raises `ConfigError` for a missing
    file, an unknown extension, a file its parser rejects, or a file whose
    top level is not a table.
    """
    path = Path(path)
    parser = _FILE_PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ConfigError(
            f"cannot read config file {path}: unknown extension {path.suffix!r}; "
            f"expected one of {', '.join(_FILE_PARSERS)}"
        )
    try:
        values = parser(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(
            f"cannot read config file {path}: {error.strerror}"
        ) from error
    except (ValueError, yaml.YAMLError) as error:  # tomllib/json errors are ValueErrors
        raise ConfigError(f"cannot parse config file {path}: {error}") from error
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ConfigError(
            f"config file {path} must hold a table of keys at the top level, "
            f"not a {type(values).__name__}"
        )
    return values


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """`base` overlaid with `update`, recursing where both hold a dict."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dotted_paths(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Every field of the tree as a dotted path, sections included.

    A section inside a dict or list is not descended into: the CLI addresses
    such a field only as a whole, with a JSON value.
    """
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        yield path
        section, inside_collection = _section_of(field.annotation)
        if section is not None and not inside_collection:
            yield from _dotted_paths(section, f"{path}.")


def _describe_unknown(key: str, candidates: Sequence[str]) -> str:
    message = f"unknown config key {key!r}"
    closest = difflib.get_close_matches(key, candidates, n=1)
    if closest:
        message += f"; did you mean {closest[0]!r}?"
    return message


def _check_override_names(model: type[BaseModel], cli_overrides: Sequence[str]) -> None:
    """Reject option tokens that address no field, before argparse sees them.

    argparse would reject them too, but its message names neither the dotted
    path nor a neighbour, it stops at the first one, and on `-h` it prints
    help and exits the process. Only tokens in option position are checked:
    the token after `--name` (no `=`) is its value.
    """
    valid = list(_dotted_paths(model))
    unknown = []
    expecting_value = False
    for token in cli_overrides:
        if expecting_value:
            expecting_value = False
        elif token.startswith("--"):
            name, separator, _ = token[2:].partition("=")
            expecting_value = not separator
            if name not in valid:
                unknown.append(_describe_unknown(name, valid))
        elif token.startswith("-"):
            unknown.append(
                f"unknown config option {token!r}; overrides are written --key value"
            )
    if unknown:
        raise ConfigError("\n".join(unknown))


def _unknown_key_messages(model: type[BaseModel], error: ValidationError) -> list[str]:
    """One message per `extra_forbidden` error, with the neighbour at that level.

    The error location is walked against the schema: a field name moves into
    its section, a dict key or list index keeps the section, and the tag
    pydantic inserts for a `Section | scalar` field (the class name) is
    skipped and left out of the printed path.
    """
    messages = []
    for item in error.errors():
        if item["type"] != "extra_forbidden":
            continue
        *location, key = (str(part) for part in item["loc"])
        section: type[BaseModel] | None = model
        inside_collection = False
        names = []
        for part in location:
            if section is not None and part == section.__name__:
                continue
            names.append(part)
            if inside_collection:
                inside_collection = False
            elif section is not None and part in section.model_fields:
                section, inside_collection = _section_of(
                    section.model_fields[part].annotation
                )
            else:
                section = None
        candidates = list(section.model_fields) if section is not None else []
        prefix = "".join(f"{name}." for name in names)
        messages.append(
            _describe_unknown(f"{prefix}{key}", [f"{prefix}{c}" for c in candidates])
        )
    return messages
