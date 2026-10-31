"""Bind numeric narrative references to consumed facts, not to model-copied values.

This is a reference/format contract, not natural-language entailment or causal validation.
Spelled-out quantities and qualitative statements remain unverified interpretations.
"""

import math
import re
from copy import deepcopy

from pydantic import ValidationError

from gpu_os_copilot.schemas import AgentDecision

REFERENCE = re.compile(r"\{\{(metric|percent|ratio):([A-Za-z0-9_.:-]+)\}\}")
COUNT_UNITS = {"pairs", "batches", "events"}
INSTRUCTION = (
    "In final narrative fields, use qualitative prose and numeric references instead of "
    "copying numbers. {{metric:NAME}} inserts the consumed catalog value and unit; "
    "{{percent:NAME}} converts a fraction to percent; "
    "{{ratio:NUMERATOR_REF:DENOMINATOR_REF}} inserts counts with a common unit and scope. "
    "Use only available evidence_catalog.metrics. Do not reconstruct counts, round values, "
    "write literal numerals or spell quantities out to bypass this contract. Keep predictions "
    "qualitative; bounded experiment parameters belong in tool arguments. Other tool statistics "
    "remain in the verification report and need not be repeated in prose. References verify "
    "numbers, not the truth of an interpretation or causal explanation."
)


def narrative_fields(value):
    """Yield only model-authored prose; numeric budgets, metadata and IDs are not prose."""
    yield value, "rationale", "rationale"
    for index, hypothesis in enumerate(value["hypotheses"]):
        for field in ("statement", "counterevidence_note", "testable_prediction"):
            yield hypothesis, field, f"hypotheses.{index}.{field}"
        for item in range(len(hypothesis["missing_evidence"])):
            yield (
                hypothesis["missing_evidence"],
                item,
                f"hypotheses.{index}.missing_evidence.{item}",
            )
    if value["selected_experiment"]:
        yield value["selected_experiment"], "justification", "selected_experiment.justification"
    for index, deferral in enumerate(value["deferred_evidence"]):
        yield deferral, "rationale", f"deferred_evidence.{index}.rationale"


def format_number(value):
    # Counts remain exact; other facts use a fixed display precision, never model rounding.
    return str(value) if isinstance(value, int) else format(value, ".12g")


def resolve_reference(kind, names, metrics, default_scope):
    refs = names.split(":")
    if len(refs) != (2 if kind == "ratio" else 1):
        return None, "MALFORMED_NUMERIC_REFERENCE"
    facts = []
    for ref in refs:
        if ref not in metrics:
            return None, "UNKNOWN_NUMERIC_METRIC"
        metric = metrics[ref]
        number = metric.get("value")
        if (
            metric.get("available") is not True
            or metric.get("availability_state") in {"unread", "unavailable"}
            or type(number) not in {int, float}
            or not math.isfinite(number)
        ):
            return None, "UNAVAILABLE_NUMERIC_METRIC"
        facts.append(
            {
                "metric_ref": ref,
                "value": number,
                "unit": metric["unit"],
                "scope": metric.get("scope", default_scope),
            }
        )
    first = facts[0]
    if kind == "percent":
        if first["unit"] != "fraction" or not math.isfinite(first["value"] * 100):
            return None, "INVALID_NUMERIC_UNIT"
        rendered = format_number(first["value"] * 100) + "%"
    elif kind == "ratio":
        second = facts[1]
        if (
            first["unit"] not in COUNT_UNITS
            or first["unit"] != second["unit"]
            or first["scope"] != second["scope"]
            or any(fact["value"] != int(fact["value"]) for fact in facts)
            or not 0 <= first["value"] <= second["value"]
            or second["value"] <= 0
        ):
            return None, "INVALID_NUMERIC_RATIO"
        rendered = f"{int(first['value'])}/{int(second['value'])} {first['unit']}"
    else:
        rendered = format_number(first["value"]) + " " + first["unit"]
    return {"kind": kind, "facts": facts, "rendered": rendered}, None


def ground_narrative(decision: AgentDecision, catalog: dict, *, visible_ids=()):
    """Return a rendered copy only when every numeric reference/literal passes checks."""
    original = decision.model_dump()
    rendered = deepcopy(original)
    metrics = catalog["metrics"]
    # Known evidence identifiers can include digits without asserting a measurement.
    known = {name for name in [*metrics, *visible_ids] if any(c.isnumeric() for c in name)}
    identifiers = (
        re.compile(
            r"(?<![\w.-])(?:" + "|".join(re.escape(n) for n in sorted(known)) + r")(?![\w.-])"
        )
        if known
        else None
    )
    errors, references = [], []
    for owner, key, path in narrative_fields(rendered):
        text = owner[key]

        def replace(match, path=path):
            result, error = resolve_reference(
                *match.groups(), metrics, catalog.get("scope", "Input trace measurement window")
            )
            if error:
                errors.append({"field": path, "code": error, "reference": match.group()})
                return match.group()
            references.append({"field": path, "reference": match.group(), **result})
            return result["rendered"]

        owner[key] = REFERENCE.sub(replace, text)
        rest = REFERENCE.sub("", text)
        if "{{" in rest or "}}" in rest:
            errors.append({"field": path, "code": "MALFORMED_NUMERIC_REFERENCE"})
        if identifiers:
            rest = identifiers.sub("", rest)
        if any(char.isnumeric() for char in rest):
            errors.append({"field": path, "code": "NUMERIC_LITERAL_REQUIRES_REFERENCE"})
    result = None
    if not errors:
        try:
            result = AgentDecision.model_validate(rendered)
        except ValidationError:
            errors.append({"code": "NUMERIC_RENDER_EXCEEDS_FIELD_LIMIT"})
    assessment = {
        "version": "numeric-grounding-v1",
        "status": "invalid" if errors else "references_resolved" if references else "no_references",
        "errors": errors,
        "references": references,
        "narrative_semantics": "not_verified",
        "input_confidence": catalog.get("input_confidence"),
        "limitations": [
            "Checks numeric references and numeral literals, not all natural-language claims.",
            "Spelled-out quantities and qualitative comparisons are not semantically validated.",
            "Correct numbers do not prove relevance, causality, independence or generalization.",
            "Values inherit input provenance; grounding cannot turn simulation into measurement.",
        ],
    }
    # Do not produce a partially corrected decision when another numeric assertion is invalid.
    return result, assessment
