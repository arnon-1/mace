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


def _reject_set_fields(model: type[BaseModel]) -> None:
    """Fail at class definition for a field typed with a set at any depth."""

    def holds_set(annotation: Any) -> bool:
        origin = get_origin(annotation) or annotation
        if origin in (set, frozenset):
            return True
        return any(holds_set(arg) for arg in get_args(annotation))

    for name, field in model.model_fields.items():
        if holds_set(field.annotation):
            raise TypeError(
                f"{model.__name__}.{name} is typed as a set; set order is not "
                f"stable across runs, so the resolved config would not be a "
                f"fixed point. Use a list."
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
        _reject_set_fields(cls)


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
        the wrong type.
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
        input had. Writing the result to any of the three file formats and
        loading it back gives an identical config, and resolving that gives an
        identical dict: the export is a fixed point. (TOML has no null, so a
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


def _section_members(model: type[BaseModel], name: str) -> tuple[type[BaseModel], ...]:
    """The section classes field `name` can hold: one for `Section` or
    `Section | None`, several for a union of sections, none for anything else."""
    field = model.model_fields.get(name)
    if field is None:
        return ()
    annotation = field.annotation
    members = (
        get_args(annotation)
        if get_origin(annotation) in (Union, UnionType)
        else (annotation,)
    )
    return tuple(m for m in members if isinstance(m, type) and issubclass(m, BaseModel))


def _dotted_paths(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Every field of the tree as a dotted path, sections included.

    A union of several sections is a leaf here: the CLI addresses it only as a
    whole, so the paths below it are not override targets.
    """
    for name in model.model_fields:
        path = f"{prefix}{name}"
        yield path
        members = _section_members(model, name)
        if len(members) == 1:
            yield from _dotted_paths(members[0], f"{path}.")


def _describe_unknown(key: str, candidates: Sequence[str]) -> str:
    message = f"unknown config key {key!r}"
    closest = difflib.get_close_matches(key, candidates, n=1)
    if closest:
        message += f"; did you mean {closest[0]!r}?"
    return message


def _check_override_names(model: type[BaseModel], cli_overrides: Sequence[str]) -> None:
    """Reject `--name` tokens that address no field, before argparse sees them.

    argparse would reject them too, but its message names neither the dotted
    path nor a neighbour, and it stops at the first one. Only tokens in option
    position are checked: the token after `--name` (no `=`) is its value.
    """
    valid = list(_dotted_paths(model))
    unknown = []
    expecting_value = False
    for token in cli_overrides:
        if expecting_value or not token.startswith("--"):
            expecting_value = False
            continue
        name, _, inline_value = token[2:].partition("=")
        expecting_value = not inline_value
        if name not in valid:
            unknown.append(_describe_unknown(name, valid))
    if unknown:
        raise ConfigError("\n".join(unknown))


def _unknown_key_messages(model: type[BaseModel], error: ValidationError) -> list[str]:
    """One message per `extra_forbidden` error, with the neighbour at that level.

    Pydantic's error location interleaves union member names with the field
    names when a section is a union of sections (`either`, `A`, `z`); those
    tags pick the member to search in and are left out of the printed path.
    """
    messages = []
    for item in error.errors():
        if item["type"] != "extra_forbidden":
            continue
        *location, key = (str(part) for part in item["loc"])
        section: type[BaseModel] | None = model
        members: tuple[type[BaseModel], ...] = ()
        names = []
        for part in location:
            tagged = [m for m in members if m.__name__ == part]
            if tagged:  # a union tag, not a key
                section, members = tagged[0], ()
                continue
            names.append(part)  # a field, or the key of a dict-valued field
            members = _section_members(section, part) if section is not None else ()
            section = members[0] if len(members) == 1 else None
            if len(members) == 1:
                members = ()
        candidates = list(section.model_fields) if section is not None else []
        prefix = "".join(f"{name}." for name in names)
        messages.append(
            _describe_unknown(f"{prefix}{key}", [f"{prefix}{c}" for c in candidates])
        )
    return messages
