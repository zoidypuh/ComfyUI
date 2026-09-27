"""Schema and chat-request checks for the pictured CLIProxy LLM node."""

import asyncio

from comfy_api_nodes.apis.openrouter import OpenRouterChatResponse, OpenRouterChoice, OpenRouterResponseMessage

import CLIProxy
from CLIProxy import nodes_openrouter


def _effort_input(schema):
    matches = [item for item in schema.inputs if item.id == "effort"]
    assert len(matches) == 1
    return matches[0]


def test_llm_schema_name_and_effort_string():
    schema = CLIProxy.CLIOpenRouterLLM.define_schema()
    assert schema.display_name == "LLM"
    effort = _effort_input(schema)
    assert (effort.display_name or effort.id) == "effort"
    assert effort.get_io_type() == "STRING"
    assert type(effort).__name__ != "Combo"
    widget = (effort.as_dict().get("widgetType") or effort.get_io_type()).lower()
    assert "combo" not in widget

    partner = nodes_openrouter.OpenRouterLLMNode.define_schema()
    assert partner.display_name == "OpenRouter LLM"
    assert all(item.id != "effort" for item in partner.inputs)


def _response():
    return OpenRouterChatResponse(
        choices=[OpenRouterChoice(message=OpenRouterResponseMessage(content="ok"))]
    )


def _run_execute(**kwargs):
    captured = {}

    async def sync_op(cls, endpoint, **call):
        captured["request"] = call["data"]
        captured["endpoint"] = endpoint
        return _response()

    original = nodes_openrouter.sync_op
    nodes_openrouter.sync_op = sync_op
    try:
        result = asyncio.run(
            CLIProxy.CLIOpenRouterLLM.execute(
                endpoint="http://winpc-1.tailed34e0.ts.net:8317/v1",
                api_key="comfy",
                prompt="describe the picture",
                model="mtplx/abliterated",
                seed=1800379687,
                **kwargs,
            )
        )
    finally:
        nodes_openrouter.sync_op = original
        nodes_openrouter._MODELS_BY_SLUG.pop("mtplx/abliterated", None)
    return captured["request"], result


def test_effort_xhigh_is_forwarded_and_blank_omits_reasoning():
    high, _ = _run_execute(effort="xhigh")
    assert high.model == "mtplx/abliterated"
    assert high.seed == 1800379687
    assert high.reasoning is not None
    assert high.reasoning.effort == "xhigh"
    user = high.messages[-1]
    assert user.content == "describe the picture"

    blank, _ = _run_execute(effort="")
    assert blank.reasoning is None
    assert blank.model == "mtplx/abliterated"

    whitespace, _ = _run_execute(effort="   ")
    assert whitespace.reasoning is None

    typed, _ = _run_execute(effort="max")
    assert typed.reasoning.effort == "max"


def test_default_execute_keeps_model_prompt_and_seed():
    request, output = _run_execute()
    assert request.model == "mtplx/abliterated"
    assert request.seed == 1800379687
    assert request.messages[-1].content == "describe the picture"
    assert request.reasoning is None
    assert output is not None
