"""ClipProxy versions of the Comfy API image, video, and LLM nodes."""

import json

from . import nodes_grok, nodes_openai, nodes_openrouter
from .runtime import Redirected, clone_schema, install_redirect, model_options

for _module in (nodes_grok, nodes_openai, nodes_openrouter):
    install_redirect(_module)


class CLIGrokImage(Redirected, nodes_grok.GrokImageNode):
    @classmethod
    def define_schema(cls):
        return clone_schema(
            nodes_grok.GrokImageNode,
            "CLIProxyGrokImage",
            "CLIProxy/image/Grok",
            replace_model_options=model_options(("grok-imagine-image",), ["grok-imagine-image-2.0"]),
        )

    @classmethod
    async def execute(cls, endpoint, api_key, **kwargs):
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_grok.GrokImageNode.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)


class CLIGrokImageEdit(Redirected, nodes_grok.GrokImageEditNodeV2):
    @classmethod
    def define_schema(cls):
        return clone_schema(
            nodes_grok.GrokImageEditNodeV2,
            "CLIProxyGrokImageEdit",
            "CLIProxy/image/Grok",
            replace_model_options=model_options(
                ("grok-imagine-image-",),
                ["grok-imagine-image-2.0", "grok-imagine-image-quality"],
            ),
        )

    @classmethod
    async def execute(cls, endpoint, api_key, **kwargs):
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_grok.GrokImageEditNodeV2.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)


def _redirected_video(name, base, model_prefixes=None):
    return type(
        name,
        (Redirected, base),
        {
            "define_schema": classmethod(
                lambda cls, _base=base, _name=name, _prefixes=model_prefixes: clone_schema(
                    _base,
                    "CLIProxy" + _name,
                    "CLIProxy/video/Grok",
                    replace_model_options=(
                        model_options(_prefixes, ["grok-imagine-video", "grok-imagine-video-1.5"])
                        if _prefixes
                        else None
                    ),
                )
            ),
            "execute": classmethod(_video_execute(base)),
        },
    )


def _video_execute(base):
    async def execute(cls, endpoint, api_key, **kwargs):
        token = cls._with_connection(endpoint, api_key)
        try:
            return await base.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)

    return execute


CLIGrokVideo = _redirected_video(
    "GrokVideo", nodes_grok.GrokVideoNode, ("grok-imagine-video",)
)
CLIGrokVideoEdit = _redirected_video("GrokVideoEdit", nodes_grok.GrokVideoEditNode)
CLIGrokVideoReference = _redirected_video("GrokVideoReference", nodes_grok.GrokVideoReferenceNode)
CLIGrokVideoExtend = _redirected_video("GrokVideoExtend", nodes_grok.GrokVideoExtendNode)


class CLIOpenAIGPTImage(Redirected, nodes_openai.OpenAIGPTImageNodeV2):
    @classmethod
    def define_schema(cls):
        return clone_schema(
            nodes_openai.OpenAIGPTImageNodeV2,
            "CLIProxyOpenAIGPTImage",
            "CLIProxy/image/OpenAI",
            replace_model_options=model_options(
                ("gpt-image-",),
                ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "or/openai/gpt-image-2.5-sunburst", "fun/gpt-image-2.5-sunburst", "gpt-image-2", "gpt-image-1.5"],
            ),
        )

    @classmethod
    async def execute(cls, endpoint, api_key, **kwargs):
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_openai.OpenAIGPTImageNodeV2.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)


class CLIOpenRouterLLM(Redirected, nodes_openrouter.OpenRouterLLMNode):
    @classmethod
    def define_schema(cls):
        schema = clone_schema(nodes_openrouter.OpenRouterLLMNode, "CLIProxyOpenRouterLLM", "CLIProxy/text/OpenRouter")
        for index, item in enumerate(schema.inputs):
            if getattr(item, "id", None) == "model":
                schema.inputs[index] = __import__("comfy_api.latest", fromlist=["IO"]).IO.String.Input(
                    "model",
                    default="grok-4.6",
                    tooltip="ClipProxy model ID from /v1/models. Endpoint must be the /v1 root, not /chat/completions or /responses.",
                )
                break
        comfy_io = __import__("comfy_api.latest", fromlist=["IO"]).IO
        schema.inputs.extend(
            [
                comfy_io.Image.Input(
                    "image_1",
                    display_name="image 1",
                    optional=True,
                    tooltip="First optional image sent to the selected multimodal model.",
                ),
                comfy_io.Image.Input(
                    "image_2",
                    display_name="image 2",
                    optional=True,
                    tooltip="Second optional image sent to the selected multimodal model.",
                ),
            ]
        )
        return schema

    @classmethod
    async def execute(cls, endpoint, api_key, model, image_1=None, image_2=None, **kwargs):
        images = [image for image in (image_1, image_2) if image is not None]
        # The copied OpenRouter implementation validates against its curated list.
        # Add the manually entered ID as a zero-cost, text-only runtime spec.
        if model not in nodes_openrouter._MODELS_BY_SLUG:
            nodes_openrouter._MODELS_BY_SLUG[model] = nodes_openrouter._ModelSpec(
                model, "standard", 0, 0, max_images=2 if images else 0
            )
        elif images:
            spec = nodes_openrouter._MODELS_BY_SLUG[model]
            if spec.max_images < len(images):
                nodes_openrouter._MODELS_BY_SLUG[model] = nodes_openrouter._ModelSpec(
                    model, spec.profile, spec.price_in, spec.price_out, max_images=len(images)
                )
        token = cls._with_connection(endpoint, api_key)
        try:
            model_payload = {"model": model}
            if images:
                model_payload["images"] = {
                    f"image_{index}": image for index, image in enumerate(images, start=1)
                }
            return await nodes_openrouter.OpenRouterLLMNode.execute.__func__(
                cls, model=model_payload, **kwargs
            )
        finally:
            cls._reset_connection(token)


class CLIProxyJSONField:
    """Extract one string field from a JSON response produced by an LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True}),
                "field": ("STRING", {"default": "description"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "extract"
    CATEGORY = "CLIProxy/Text"

    def extract(self, json_text, field):
        raw = json_text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("LLM output is not valid JSON")
            payload = json.loads(raw[start : end + 1])
        if not isinstance(payload, dict) or field not in payload:
            raise ValueError(f"JSON field not found: {field}")
        value = payload[field]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return (str(value),)


NODE_CLASS_MAPPINGS = {
    "CLIProxyGrokImage": CLIGrokImage,
    "CLIProxyGrokImageEdit": CLIGrokImageEdit,
    "CLIProxyOpenAIGPTImage": CLIOpenAIGPTImage,
    "CLIProxyGrokVideo": CLIGrokVideo,
    "CLIProxyGrokVideoEdit": CLIGrokVideoEdit,
    "CLIProxyGrokVideoReference": CLIGrokVideoReference,
    "CLIProxyGrokVideoExtend": CLIGrokVideoExtend,
    "CLIProxyOpenRouterLLM": CLIOpenRouterLLM,
    "CLIProxyJSONField": CLIProxyJSONField,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
