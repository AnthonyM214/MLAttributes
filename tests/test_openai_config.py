import unittest

from places_attr_conflation.openai_config import (
    DEFAULT_API_SURFACE,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEXT_VERBOSITY,
    GPT55_MODEL,
    OpenAIWorkflowConfig,
    config_for_signature,
)


class OpenAIConfigTests(unittest.TestCase):
    def test_defaults_target_gpt55_responses(self) -> None:
        self.assertEqual(GPT55_MODEL, "gpt-5.5")
        self.assertEqual(DEFAULT_API_SURFACE, "responses")
        self.assertEqual(DEFAULT_REASONING_EFFORT, "medium")
        self.assertEqual(DEFAULT_TEXT_VERBOSITY, "medium")

    def test_signature_configs_are_available(self) -> None:
        config = config_for_signature("AttributeResolver")

        self.assertEqual(config.model, "gpt-5.5")
        self.assertEqual(config.api_surface, "responses")
        self.assertEqual(config.reasoning_effort, "medium")

    def test_unknown_signature_uses_default_config(self) -> None:
        self.assertEqual(config_for_signature("Unknown"), OpenAIWorkflowConfig())


if __name__ == "__main__":
    unittest.main()
