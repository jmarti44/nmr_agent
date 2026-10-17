import json
import subprocess
import sys
from pathlib import Path

from src.prompting import templates
from src.schemas import FunctionalGroup
from src.verify.shift_tables import C_RULES, H_RULES, SHIFT_TABLE_VERSION
from src.verify.smarts import SPECS

MANIFEST = Path(__file__).parent / "prompt_manifest.json"


def test_prompt_text_is_covered_by_the_version_manifest():
    """Any edit to the wording or the embedded tables changes the hash and must come with a version bump."""
    recorded = json.loads(MANIFEST.read_text())
    assert templates.FULL_VERSION in recorded, (
        f"{templates.FULL_VERSION} is not in prompt_manifest.json: bump PROMPT_VERSION or "
        f"SHIFT_TABLE_VERSION and record the new hash {templates.prompt_hash()}"
    )
    assert recorded[templates.FULL_VERSION] == templates.prompt_hash(), (
        "the system prompt changed without a version bump: bump PROMPT_VERSION (or SHIFT_TABLE_VERSION for "
        f"table changes) and record {templates.prompt_hash()} in prompt_manifest.json"
    )


def test_version_string_carries_the_shift_table_version():
    assert templates.FULL_VERSION == f"{templates.PROMPT_VERSION}+shifts.{SHIFT_TABLE_VERSION}"


def test_reference_ranges_are_generated_from_the_verifier_tables():
    text = templates.system_prompt()
    for rule in [*H_RULES.values(), *C_RULES.values()]:
        if rule.key == "h_unclassified":
            continue  # a verifier catch-all, not chemistry for the model
        assert rule.key in text, rule.key
        assert f"{rule.lo:g}-{rule.hi:g}" in text, rule.key
        if rule.confirm:
            assert f"core {rule.confirm[0]:g}-{rule.confirm[1]:g}" in text
    assert "h_unclassified" not in text


def test_every_functional_group_is_defined_with_the_checkers_smarts():
    text = templates.system_prompt()
    for g in FunctionalGroup:
        assert g.value in text
        assert SPECS[g].smarts[0] in text


def test_system_prompt_is_static_and_does_not_depend_on_the_request():
    assert templates.system_prompt() == templates.system_prompt()
    assert "{max_candidates}" not in templates.system_prompt()  # no unfilled placeholder
    assert "candidate structure(s)" not in templates.system_prompt()  # the count lives in the user message


def test_rules_state_the_positions_that_differ_from_the_spec_text():
    text = templates.system_prompt()
    assert "(observed signals - 1) and (observed signals + 2)" in text  # matches what V06 checks
    assert "may also be genuine analyte signals" in text  # impurity_suspect peaks stay usable


def test_system_prompt_size_is_reasonable():
    assert 1000 < templates.estimate_tokens(templates.system_prompt()) < 4000


def test_importing_the_prompt_package_does_not_load_any_model_sdk_or_chemistry_toolkit():
    code = (
        "import sys; import src.prompting; import src.processing.normalize;"
        "print([m for m in ('anthropic','httpx','httpx2','rdkit','nmrglue') if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_the_prompt_says_the_claims_describe_the_first_candidate_only():
    text = templates.system_prompt()
    assert "describes your first" in text and "(rank 1) candidate only" in text
