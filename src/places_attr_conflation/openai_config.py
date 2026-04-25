"""OpenAI defaults for optional model-backed workflow experiments.

The benchmark path remains deterministic and does not call OpenAI. These
constants only define the target model and safe starting defaults for future
implementations behind the typed signatures in ``signatures.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


GPT55_MODEL = "gpt-5.5"
DEFAULT_API_SURFACE = "responses"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TEXT_VERBOSITY = "medium"


@dataclass(frozen=True)
class OpenAIWorkflowConfig:
    """Minimal config contract for future GPT-backed signature adapters."""

    model: str = GPT55_MODEL
    api_surface: str = DEFAULT_API_SURFACE
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    text_verbosity: str = DEFAULT_TEXT_VERBOSITY


SIGNATURE_CONFIGS: dict[str, OpenAIWorkflowConfig] = {
    "PlacePairMatcher": OpenAIWorkflowConfig(),
    "EvidenceExtractor": OpenAIWorkflowConfig(),
    "SourceJudge": OpenAIWorkflowConfig(),
    "AttributeResolver": OpenAIWorkflowConfig(),
}


def config_for_signature(signature_name: str) -> OpenAIWorkflowConfig:
    """Return the GPT-5.5 starting config for a signature adapter."""

    return SIGNATURE_CONFIGS.get(signature_name, OpenAIWorkflowConfig())
