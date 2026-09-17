"""`ReforgeBaseConfig`: file formats, precedence, dotted overrides, unknown keys,
and the resolved export's fixed point."""

import json
import subprocess
import sys

import pytest
import yaml
from mace_core.config import ConfigError, ConfigSection, ReforgeBaseConfig
from pydantic import Field, ValidationError

# ---------------------------------------------------------------------------
# The demo schema: two levels of nesting, a list, an optional, a Literal.


class RadialSection(ConfigSection):
    num_bessel: int = 8
    cutoff: float = 5.0


class ModelSection(ConfigSection):
    num_interactions: int = 2
    hidden_irreps: str = "128x0e + 128x1o"
    radial: RadialSection = RadialSection()


class DataSection(ConfigSection):
    train_file: str | None = None
    valid_fraction: float = 0.1
    energy_key: str = "REF_energy"
    heads: list[str] = Field(default_factory=lambda: ["default"])


class StageTwoSection(ConfigSection):
    start_epoch: int = 100
    energy_weight: float = 1000.0


class DemoConfig(ReforgeBaseConfig):
    name: str = "mace"
    seed: int = 123
    default_dtype: str = "float64"
    model: ModelSection = ModelSection()
    data: DataSection = DataSection()
    #: An optional section: absent unless the file or the CLI opens it.
    stage_two: StageTwoSection | None = None


#: One config, as a dict. Each format test writes it out and loads it back.
FILE_VALUES = {
    "name": "water",
    "seed": 7,
    "model": {"num_interactions": 4, "radial": {"cutoff": 4.5}},
    "data": {"train_file": "train.xyz", "heads": ["pbe", "r2scan"]},
}


def to_toml(values, prefix=""):
    """Enough TOML for a None-free config: scalars and lists share JSON's
    literal syntax, nested dicts become `[a.b]` tables after the scalars."""
    lines = [
        f"{k} = {json.dumps(v)}" for k, v in values.items() if not isinstance(v, dict)
    ]
    for key, value in values.items():
        if isinstance(value, dict):
            lines += [f"\n[{prefix}{key}]", to_toml(value, f"{prefix}{key}.")]
    return "\n".join(lines)


def dump(values, extension):
    if extension == ".toml":
        return to_toml(values)
    if extension == ".json":
        return json.dumps(values)
    return yaml.safe_dump(values)


def write_config(tmp_path, extension, values=FILE_VALUES):
    path = tmp_path / f"config{extension}"
    path.write_text(dump(values, extension), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# File loading


@pytest.mark.parametrize("extension", [".toml", ".yaml", ".yml", ".json"])
def test_same_config_loads_identically_from_every_format(tmp_path, extension):
    config = DemoConfig.load(write_config(tmp_path, extension))
    assert config == DemoConfig.model_validate(FILE_VALUES)
    # The file set two fields at depth two; the sibling kept its default.
    assert config.model.radial.cutoff == 4.5
    assert config.model.radial.num_bessel == 8


def test_unknown_extension_is_an_error(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("seed = 1", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown extension '\.ini'"):
        DemoConfig.load(path)


def test_empty_file_is_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert DemoConfig.load(path) == DemoConfig()


def test_file_must_be_a_table_at_the_top(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="table of keys at the top level"):
        DemoConfig.load(path)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"cannot read config file .*nope\.yaml"):
        DemoConfig.load(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("extension", "text"),
    [(".toml", "seed = \n"), (".yaml", "seed: [1\n"), (".json", "{")],
)
def test_malformed_file_is_a_config_error(tmp_path, extension, text):
    path = tmp_path / f"broken{extension}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"cannot parse config file .*broken"):
        DemoConfig.load(path)


# ---------------------------------------------------------------------------
# Precedence: defaults < file < CLI. The legacy behaviour this pins is
# tests/unit/test_arg_parser.py::test_cli_flag_overrides_yaml_config.


def test_no_inputs_gives_the_defaults():
    config = DemoConfig.load()
    assert config == DemoConfig()
    assert config.model.num_interactions == 2


def test_file_overrides_defaults(tmp_path):
    config = DemoConfig.load(write_config(tmp_path, ".yaml"))
    assert config.model.num_interactions == 4  # from the file
    assert config.default_dtype == "float64"  # untouched default


def test_cli_overrides_file_which_overrides_defaults(tmp_path):
    config = DemoConfig.load(
        write_config(tmp_path, ".toml"), ["--model.num_interactions", "3"]
    )
    assert config.model.num_interactions == 3  # CLI beats the file's 4
    assert config.model.radial.cutoff == 4.5  # the file's other values survive
    assert config.seed == 7
    assert config.model.radial.num_bessel == 8  # defaults fill the rest
    assert config.default_dtype == "float64"


# ---------------------------------------------------------------------------
# Dotted CLI overrides


def test_dotted_override_reaches_a_two_level_nested_field():
    config = DemoConfig.load(cli_overrides=["--model.radial.cutoff", "6.0"])
    assert config.model.radial.cutoff == 6.0
    assert config.model.radial.num_bessel == 8


def test_dotted_override_opens_an_optional_section():
    config = DemoConfig.load(cli_overrides=["--stage_two.start_epoch", "50"])
    assert config.stage_two == StageTwoSection(start_epoch=50)
    assert DemoConfig.load().stage_two is None


def test_override_forms_and_types():
    config = DemoConfig.load(
        cli_overrides=[
            "--seed=9",
            "--data.train_file",
            "null",
            "--data.heads",
            "a",
            "--data.heads",
            "b",
        ]
    )
    assert config.seed == 9
    assert config.data.train_file is None
    assert config.data.heads == ["a", "b"]


def test_override_of_the_wrong_type_is_a_validation_error():
    with pytest.raises(ValidationError, match="seed"):
        DemoConfig.load(cli_overrides=["--seed", "seven"])


def test_override_missing_its_value_is_a_config_error():
    with pytest.raises(ConfigError, match="cannot parse CLI overrides"):
        DemoConfig.load(cli_overrides=["--seed"])


def test_value_starting_with_dashes_needs_the_equals_form():
    assert DemoConfig.load(cli_overrides=["--name=--odd"]).name == "--odd"
    # Split off, argparse takes it for an option; the error names `--name`,
    # not a spurious unknown key `odd`.
    with pytest.raises(ConfigError, match="argument --name: expected one argument"):
        DemoConfig.load(cli_overrides=["--name", "--odd"])


def test_dict_valued_field_takes_json_and_is_not_dotted_into():
    class Sources(ReforgeBaseConfig):
        by_name: dict[str, RadialSection] = Field(default_factory=dict)

    config = Sources.load(cli_overrides=["--by_name", '{"pbe": {"cutoff": 4.0}}'])
    assert config.by_name == {"pbe": RadialSection(cutoff=4.0)}
    with pytest.raises(ConfigError, match=r"unknown config key 'by_name\.pbe\.cutoff'"):
        Sources.load(cli_overrides=["--by_name.pbe.cutoff", "4.0"])
    with pytest.raises(ConfigError, match=r"unknown config key 'by_name\.pbe\.cutof'"):
        Sources.load(cli_overrides=["--by_name", '{"pbe": {"cutof": 4.0}}'])


# ---------------------------------------------------------------------------
# Unknown keys name the key and its nearest neighbour, in files and on the CLI.


def test_unknown_top_level_key_in_file(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("sead: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"'sead'; did you mean 'seed'\?"):
        DemoConfig.load(path)


def test_unknown_nested_key_in_file_names_the_dotted_path(tmp_path):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"model": {"radial": {"cutof": 4.0}}}), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match=r"'model\.radial\.cutof'; did you mean 'model\.radial\.cutoff'\?",
    ):
        DemoConfig.load(path)


def test_every_unknown_key_is_reported_at_once(tmp_path):
    path = tmp_path / "typos.yaml"
    path.write_text("sead: 1\nmodel:\n  num_interaction: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        DemoConfig.load(path)
    assert "'sead'" in str(excinfo.value)
    assert "'model.num_interaction'" in str(excinfo.value)


def test_unknown_key_without_a_close_neighbour_still_names_it(tmp_path):
    path = tmp_path / "far.yaml"
    path.write_text("zzzzzz: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown config key 'zzzzzz'$"):
        DemoConfig.load(path)


def test_unknown_dotted_override_names_the_neighbour():
    with pytest.raises(
        ConfigError,
        match=r"'model\.num_interaction'; did you mean 'model\.num_interactions'\?",
    ):
        DemoConfig.load(cli_overrides=["--model.num_interaction", "3"])


def test_unknown_key_inside_an_optional_section():
    with pytest.raises(
        ConfigError,
        match=r"'stage_two\.start'; did you mean 'stage_two\.start_epoch'\?",
    ):
        DemoConfig.load(cli_overrides=["--stage_two.start", "50"])


def test_unknown_key_inside_a_union_of_sections_names_the_member(tmp_path):
    class Either(ReforgeBaseConfig):
        either: RadialSection | StageTwoSection = RadialSection()

    path = tmp_path / "union.json"
    path.write_text('{"either": {"cutoff": 4.0, "cutof": 4.0}}', encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        Either.load(path)
    # One message per member pydantic tried; the member's class name is not a key.
    assert "'either.cutof'; did you mean 'either.cutoff'?" in str(excinfo.value)
    assert "RadialSection" not in str(excinfo.value)


def test_direct_construction_rejects_unknown_keys_too():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DemoConfig(model={"num_interaction": 3})


# ---------------------------------------------------------------------------
# Resolved export


def test_resolved_dict_has_every_default_in_declaration_order(tmp_path):
    # The file lists keys in the reverse of the schema's order.
    path = tmp_path / "reversed.yaml"
    path.write_text("seed: 1\nname: x\n", encoding="utf-8")
    resolved = DemoConfig.load(path).to_resolved_dict()
    assert list(resolved) == [
        "name",
        "seed",
        "default_dtype",
        "model",
        "data",
        "stage_two",
    ]
    assert resolved["stage_two"] is None
    assert list(resolved["model"]) == ["num_interactions", "hidden_irreps", "radial"]
    assert resolved["model"]["radial"] == {"num_bessel": 8, "cutoff": 5.0}
    assert resolved["data"]["train_file"] is None


def assert_fixed_point(tmp_path, first, extension):
    written = tmp_path / f"resolved{extension}"
    written.write_text(dump(first, extension), encoding="utf-8")
    second = DemoConfig.load(written).to_resolved_dict()
    assert second == first
    assert json.dumps(second) == json.dumps(first)  # order included


@pytest.mark.parametrize("extension", [".yaml", ".json"])
def test_file_to_resolved_to_file_to_resolved_is_a_fixed_point(tmp_path, extension):
    first = DemoConfig.load(
        write_config(tmp_path, ".toml"), ["--model.num_interactions", "3"]
    ).to_resolved_dict()
    assert first["stage_two"] is None  # a None is part of what has to survive
    assert_fixed_point(tmp_path, first, extension)


def test_fixed_point_holds_through_toml_when_nothing_is_none(tmp_path):
    # TOML has no null, so the optional section is opened and the optional
    # file name set; the resolved dict then goes through all three formats.
    first = DemoConfig.load(
        write_config(tmp_path, ".yaml"), ["--stage_two.start_epoch", "50"]
    ).to_resolved_dict()
    assert "null" not in json.dumps(first)
    for extension in (".toml", ".yaml", ".json"):
        assert_fixed_point(tmp_path, first, extension)


def test_set_fields_are_rejected_at_class_definition():
    # A set's iteration order changes with the hash seed, so its resolved
    # export could not be a fixed point across runs.
    with pytest.raises(TypeError, match=r"Bad\.tags is typed as a set.*Use a list"):

        class Bad(ConfigSection):
            tags: list[set[str]] = Field(default_factory=list)


def test_user_dict_holds_only_what_was_set(tmp_path):
    config = DemoConfig.load(
        write_config(tmp_path, ".json"), ["--model.num_interactions", "3"]
    )
    assert config.to_user_dict() == {
        "name": "water",
        "seed": 7,
        "model": {"num_interactions": 3, "radial": {"cutoff": 4.5}},
        "data": {"train_file": "train.xyz", "heads": ["pbe", "r2scan"]},
    }


# ---------------------------------------------------------------------------
# Nothing but the file and the CLI feeds a config.


def test_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("NAME", "from-the-environment")
    monkeypatch.setenv("SEED", "99")
    config = DemoConfig.load()
    assert config.name == "mace"
    assert config.seed == 123


def test_config_module_imports_neither_torch_nor_jax():
    """In a fresh interpreter, so another test's imports cannot mask a leak."""
    code = (
        "import sys, mace_core.config; "
        "leaked = {'torch', 'jax', 'e3nn'} & set(sys.modules); "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
