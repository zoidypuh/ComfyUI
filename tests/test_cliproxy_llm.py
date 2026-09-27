"""CLIProxy LLM node: visible schema and multimodal execute path."""

import asyncio
import os
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from custom_nodes.CLIProxy import nodes_openrouter
from custom_nodes.CLIProxy.__init__ import CLIOpenRouterLLM
from comfy_api_nodes.apis.openrouter import OpenRouterChatResponse, OpenRouterChoice, OpenRouterResponseMessage


def _widget_labels(schema):
    labels = []
    for item in schema.inputs:
        if getattr(item, "io_type", None) == "IMAGE":
            continue
        labels.append(getattr(item, "display_name", None) or item.id)
    return labels


def _image(seed: int) -> torch.Tensor:
    value = (seed % 7 + 1) / 10
    return torch.full((1, 4, 4, 3), value, dtype=torch.float32)


class CLIProxyLLMSchemaTest(unittest.TestCase):
    def test_define_schema_layout(self):
        schema = CLIOpenRouterLLM.define_schema()
        self.assertEqual(schema.node_id, "CLIProxyOpenRouterLLM")
        self.assertEqual(schema.display_name, "CLIProxy LLM")

        images = [item for item in schema.inputs if getattr(item, "io_type", None) == "IMAGE"]
        self.assertEqual(len(images), 3)
        self.assertEqual([item.id for item in images], ["image_1", "image_2", "image_3"])
        self.assertEqual(
            [item.display_name for item in images],
            ["image 1", "image 2", "image 3"],
        )
        self.assertTrue(all(item.optional for item in images))

        labels = _widget_labels(schema)
        model_at = labels.index("model")
        seed_at = labels.index("seed")
        self.assertEqual(labels[model_at + 1], "api key")
        self.assertEqual(labels[seed_at + 1], "system prompt")

        by_id = {item.id: item for item in schema.inputs}
        api_key = by_id["api_key"]
        system_prompt = by_id["system_prompt"]
        self.assertEqual(api_key.default, "comfy")
        self.assertFalse(api_key.advanced)
        self.assertEqual(api_key.display_name, "api key")
        self.assertFalse(system_prompt.advanced)
        self.assertEqual(system_prompt.display_name, "system prompt")
        self.assertTrue(by_id["endpoint"].advanced)


class CLIProxyLLMExecuteTest(unittest.TestCase):
    def _run(self, **kwargs):
        captured = {}

        async def fake_sync(cls, endpoint, response_model=None, data=None, **_kwargs):
            captured["data"] = data
            captured["endpoint"] = endpoint
            return OpenRouterChatResponse(
                choices=[OpenRouterChoice(message=OpenRouterResponseMessage(content="ok"))]
            )

        async def go():
            with patch.object(nodes_openrouter, "sync_op", fake_sync):
                return await CLIOpenRouterLLM.execute(
                    endpoint="http://winpc-1.tailed34e0.ts.net:8317/v1",
                    api_key="comfy",
                    model="unknown-cliproxy-model",
                    prompt="describe these",
                    seed=0,
                    **kwargs,
                )

        result = asyncio.run(go())
        self.assertEqual(result.args[0], "ok")
        return captured["data"]

    def test_three_images_and_system_prompt(self):
        request = self._run(
            image_1=_image(1),
            image_2=_image(2),
            image_3=_image(3),
            system_prompt="be brief",
        )
        messages = request.messages
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "be brief")
        user = messages[1]
        self.assertEqual(user.role, "user")
        image_blocks = [block for block in user.content if block.type == "image_url"]
        self.assertEqual(len(image_blocks), 3)
        self.assertTrue(all(block.image_url.url.startswith("data:image/png;base64,") for block in image_blocks))
        self.assertEqual(len({block.image_url.url for block in image_blocks}), 3)

    def test_one_image_empty_system_prompt(self):
        request = self._run(image_1=_image(4), system_prompt="")
        roles = [message.role for message in request.messages]
        self.assertNotIn("system", roles)
        user = request.messages[0]
        self.assertEqual(user.role, "user")
        image_blocks = [block for block in user.content if block.type == "image_url"]
        self.assertEqual(len(image_blocks), 1)


if __name__ == "__main__":
    unittest.main()
