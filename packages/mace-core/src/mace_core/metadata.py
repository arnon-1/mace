"""The record every v1 model carries about how it was made.

`ModelMetadata` is stored alongside the weights of every trained model, not
only foundation models. It is plain data: a Pydantic tree that serialises to
JSON with `to_json()` and comes back, without loss, through `from_json()`.

The record is versioned. `SCHEMA_VERSION` is bumped whenever a field is
added, removed or changes meaning, and `from_json()` refuses a record written
under a version this code does not know, so a newer checkpoint fails loudly
at load time instead of being read with the wrong meanings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from mace_core.config import ReforgeBaseConfig

__all__ = [
    "SCHEMA_VERSION",
    "Citation",
    "ConfigRecord",
    "DataSummary",
    "E0Details",
    "MetadataSchemaError",
    "ModelMetadata",
    "Provenance",
    "format_citations",
]

#: The schema version this module writes and the only one it reads. Bump it
#: together with the `Literal` on `ModelMetadata.schema_version`.
SCHEMA_VERSION: Final = 1


class MetadataSchemaError(ValueError):
    """The metadata was written under a schema version this code cannot read."""


class _Record(BaseModel):
    """Common ground: unknown keys are errors, so a typo cannot be stored."""

    model_config = ConfigDict(extra="forbid")


class ConfigRecord(_Record):
    """The training configuration, as written and as resolved.

    Both are the JSON-native dicts a `ReforgeBaseConfig` exports: `user` is
    `to_user_dict()`, the keys the config file and the command line set, and
    `resolved` is `to_resolved_dict()`, every key with defaults filled in.
    Build it with `from_config()` so the two cannot be mixed up.
    """

    user: dict[str, Any] = Field(default_factory=dict)
    resolved: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, config: ReforgeBaseConfig) -> ConfigRecord:
        return cls(user=config.to_user_dict(), resolved=config.to_resolved_dict())


class Provenance(_Record):
    """Which code produced the model."""

    #: Version of the `mace-core` distribution, as `mace_core.__version__` reports it.
    code_version: str
    #: Full hash of the commit the code was run from; None when not in a checkout.
    git_commit: str | None = None


class DataSummary(_Record):
    """Automated summary of the fitting data.

    Reference-quantity keys name the method that produced the reference as a
    prefix on the quantity: `pbe_energy`, `pbe_forces`, `r2scan_energy`.
    `reference_keys` lists the keys the model was fitted to, under that
    convention, so a reader can tell which level of theory a model reproduces.
    """

    #: Files or dataset names the fitting data came from.
    sources: list[str] = Field(default_factory=list)
    num_configurations: int | None = None
    num_atoms: int | None = None
    #: Chemical symbols of every element present.
    elements: list[str] = Field(default_factory=list)
    reference_keys: list[str] = Field(default_factory=list)


class E0Details(_Record):
    """How the per-element reference energies (the E0s) were obtained.

    `values` maps chemical symbol to E0 in the model's energy unit. Symbols
    rather than atomic numbers, because JSON keys are strings and an integer
    key would not survive the round trip.
    """

    #: "explicit" when the E0s were given; "estimated" when fitted from the data.
    source: Literal["explicit", "estimated"]
    #: The estimation method (e.g. "average", "least_squares"); None when explicit.
    method: str | None = None
    #: Parameters of the method or reference, e.g. which data the fit used.
    parameters: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, float] = Field(default_factory=dict)


class Citation(_Record):
    """One work users of the model are asked to cite."""

    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None


class ModelMetadata(_Record):
    """The mandatory per-model record. See the module docstring."""

    #: Pinned to the version this code reads; a bump here is a schema change.
    schema_version: Literal[1] = SCHEMA_VERSION
    config: ConfigRecord = Field(default_factory=ConfigRecord)
    provenance: Provenance
    data: DataSummary = Field(default_factory=DataSummary)
    e0: E0Details | None = None
    #: DOI of the model itself, not of the papers describing it.
    doi: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ModelMetadata:
        """Parse a record written by `to_json()`.

        Raises `MetadataSchemaError` when the record carries a schema version
        this code does not read, before any field is interpreted.
        """
        version = json.loads(text).get("schema_version")
        if version != SCHEMA_VERSION:
            hint = (
                "it was written by a newer mace-core; upgrade to read it"
                if isinstance(version, int) and version > SCHEMA_VERSION
                else "no migration exists for it"
            )
            raise MetadataSchemaError(
                f"model metadata has schema_version {version!r}, but this "
                f"mace-core reads schema_version {SCHEMA_VERSION}: {hint}"
            )
        return cls.model_validate_json(text)


def format_citations(citations: Iterable[Citation]) -> str:
    """Render citations as a numbered, printable block; empty for none.

    One line per citation: authors, title, venue and year, then the DOI or
    URL. Fields that are unset are left out rather than printed as None.
    """
    lines = []
    for number, citation in enumerate(citations, start=1):
        parts = []
        if citation.authors:
            parts.append(", ".join(citation.authors))
        parts.append(citation.title)
        if citation.venue and citation.year:
            parts.append(f"{citation.venue} ({citation.year})")
        elif citation.venue or citation.year:
            parts.append(str(citation.venue or citation.year))
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)
        lines.append(f"[{number}] " + ". ".join(parts))
    return "\n".join(lines)
