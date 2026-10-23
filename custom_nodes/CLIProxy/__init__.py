"""ClipProxy versions of the Comfy API image, video, and LLM nodes."""

import copy
import json

from . import nodes_grok, nodes_meta, nodes_openai, nodes_openrouter
from .runtime import Redirected, clone_schema, install_redirect, model_options

for _module in (nodes_grok, nodes_meta, nodes_openai, nodes_openrouter):
    install_redirect(_module)


DEFAULT_GROK_IMAGE_MODEL = "fun/grok-imagine-image-quality"
DEFAULT_GROK_IMAGE_EDIT_MODEL = "fun/grok-imagine-image-quality"
_IMAGE_MODEL_TOOLTIP = (
    "ClipProxy image model ID from /v1/models, e.g. grok-imagine-image, "
    "grok-imagine-image-quality, grok-imagine-image-2.0 or fun/grok-imagine-image."
)


class CLIGrokImage(Redirected, nodes_grok.GrokImageNode):
    @classmethod
    def define_schema(cls):
        schema = clone_schema(nodes_grok.GrokImageNode, "CLIProxyGrokImage", "CLIProxy/image/Grok")
        schema = _free_text_model(schema, DEFAULT_GROK_IMAGE_MODEL, _IMAGE_MODEL_TOOLTIP)
        return _price_badge(_GROK_IMAGE_BADGE_EXPR, ["model", "number_of_images", "resolution", "quality"])(schema)

    @classmethod
    async def execute(cls, endpoint, api_key, model, **kwargs):
        model = (model or "").strip() or DEFAULT_GROK_IMAGE_MODEL
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_grok.GrokImageNode.execute.__func__(cls, model=model, **kwargs)
        finally:
            cls._reset_connection(token)


class CLIGrokImageEdit(Redirected, nodes_grok.GrokImageEditNodeV2):
    @classmethod
    def define_schema(cls):
        schema = clone_schema(nodes_grok.GrokImageEditNodeV2, "CLIProxyGrokImageEdit", "CLIProxy/image/Grok")
        # Text model + the grok-imagine-image-2.0 option's inputs (images 1-3,
        # resolution, number_of_images, quality, aspect_ratio) as plain widgets.
        schema = _free_text_model(schema, DEFAULT_GROK_IMAGE_EDIT_MODEL, _IMAGE_MODEL_TOOLTIP)
        return _price_badge(_GROK_IMAGE_BADGE_EXPR, ["model", "number_of_images", "resolution", "quality"])(schema)

    @classmethod
    async def execute(
        cls,
        endpoint,
        api_key,
        prompt,
        model,
        seed,
        images=None,
        resolution="1K",
        number_of_images=1,
        quality="medium",
        aspect_ratio="auto",
    ):
        model_payload = {
            "model": (model or "").strip() or DEFAULT_GROK_IMAGE_EDIT_MODEL,
            "images": images or {},
            "resolution": resolution,
            "number_of_images": number_of_images,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
        }
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_grok.GrokImageEditNodeV2.execute.__func__(
                cls, prompt=prompt, model=model_payload, seed=seed
            )
        finally:
            cls._reset_connection(token)


def _grok_video_loop_inputs(schema):
    """Append last-frame / loop / image-format inputs after endpoint/api_key.

    Appending (instead of inserting next to ``image``) keeps widget order, so
    saved workflows' widgets_values still map to the same widgets.
    """
    comfy_io = __import__("comfy_api.latest", fromlist=["IO"]).IO
    schema.inputs.extend(
        [
            comfy_io.Image.Input(
                "last_image",
                optional=True,
                tooltip="Optional exact last frame (sent as last_frame). Requires 'image'.",
            ),
            comfy_io.Boolean.Input(
                "loop",
                default=False,
                optional=True,
                tooltip="Loop: send the start 'image' as last_frame too, so the video ends on its first frame (up to 720p). At 1080p (where last_frame is not allowed) the clip is generated natively and the loop is closed locally with a 1s cross-fade (output 1s shorter). Requires 'image'.",
            ),
            comfy_io.Combo.Input(
                "image_format",
                options=["png", "jpeg"],
                default="png",
                optional=True,
                tooltip="Encoding for image and last_frame. png = full-resolution lossless; jpeg = full resolution at jpeg_quality. Never resized.",
            ),
            comfy_io.Int.Input(
                "jpeg_quality",
                default=90,
                min=1,
                max=100,
                step=1,
                optional=True,
                tooltip="JPEG quality (1-100). Used only when image_format is jpeg.",
            ),
        ]
    )
    return schema


DEFAULT_GROK_VIDEO_MODEL = "fun/grok-imagine-video-1.5"


def _free_text_video_model(schema):
    """Free-text video model ID, e.g. fun/grok-imagine-video-1.5."""
    return _free_text_model(
        schema,
        DEFAULT_GROK_VIDEO_MODEL,
        "ClipProxy video model ID from /v1/models, e.g. "
        "grok-imagine-video-1.5-preview or fun/grok-imagine-video-1.5.",
    )


# Price badges for the ClipProxy nodes. These are ESTIMATES only (shown with "~").
# fun/ models are billed by apikey.fun: video 1.5 ~$0.075/s (Gismar's apikey.fun rate),
# images ~$0.10 (1K) / $0.15 (2K) per image (apikey.fun public Grok group list).
# Other models fall back to the upstream Comfy partner-node table (xAI list price,
# x1.43 Comfy markup for 1.5) - xAI OAuth/subscription accounts are not billed per call.
_GROK_VIDEO_BADGE_EXPR = """
(
  $m := widgets.model;
  $isFun := $substring($m, 0, 4) = "fun/";
  $is15 := $contains($m, "1.5");
  $rate := $is15
    ? (widgets.resolution = "1080p" ? 0.25 : (widgets.resolution = "720p" ? 0.14 : 0.08))
    : (widgets.resolution = "720p" ? 0.07 : 0.05);
  $imgCost := $is15 ? 0.01 : 0.002;
  $base := $rate * widgets.duration;
  $total := inputs.image.connected ? $base + $imgCost : $base;
  $usd := ($isFun and $is15) ? 0.075 * widgets.duration : ($is15 ? $total * 1.43 : $total);
  {"type":"usd","usd": $usd, "format": {"approximate": true}}
)
"""

_GROK_IMAGE_BADGE_EXPR = """
(
  $m := widgets.model;
  $b := $split($m, "/")[-1];
  $isFun := $substring($m, 0, 4) = "fun/";
  $is1k := $lowercase(widgets.resolution) != "2k";
  $n := widgets.number_of_images ? widgets.number_of_images : 1;
  $rate := $isFun
    ? ($is1k ? 0.10 : 0.15)
    : ($b = "grok-imagine-image-2.0"
        ? (widgets.quality = "low" ? ($is1k ? 0.04 : 0.06) : ($is1k ? 0.06 : 0.08))
        : ($b = "grok-imagine-image-quality"
            ? ($is1k ? 0.05 : 0.07)
            : ($contains($b, "pro") ? 0.07 : 0.02)));
  {"type":"usd","usd": $rate * $n, "format": {"approximate": true}}
)
"""


def _price_badge(expr, widgets, inputs=None):
    def _apply(schema):
        comfy_io = __import__("comfy_api.latest", fromlist=["IO"]).IO
        names = {getattr(i, "id", None) for i in schema.inputs}
        schema.price_badge = comfy_io.PriceBadge(
            depends_on=comfy_io.PriceBadgeDepends(
                widgets=[w for w in widgets if w in names], inputs=list(inputs or [])
            ),
            expr=expr,
        )
        return schema

    return _apply


def _free_text_model(schema, default, tooltip):
    """Replace the hardcoded model dropdown with a free-text model ID (same slot,
    so saved workflows keep their widget order). Any /v1/models ID works."""
    comfy_io = __import__("comfy_api.latest", fromlist=["IO"]).IO
    for index, item in enumerate(schema.inputs):
        if getattr(item, "id", None) == "model":
            new_inputs = [comfy_io.String.Input("model", default=default, tooltip=tooltip)]
            # A DynamicCombo carries nested inputs (Extend: duration; Image Edit: the
            # first option's images/resolution/...); expose them as plain inputs.
            for option in getattr(item, "options", None) or []:
                nested = getattr(option, "inputs", None)
                if nested:
                    new_inputs.extend(copy.deepcopy(i) for i in nested)
                    break
            schema.inputs[index : index + 1] = new_inputs
            break
    return schema


def _chain(*fns):
    def _apply(schema):
        for fn in fns:
            schema = fn(schema)
        return schema

    return _apply


def _redirected_video(name, base, model_prefixes=None, extend_schema=None):
    def _define(cls, _base=base, _name=name, _prefixes=model_prefixes, _extend=extend_schema):
        schema = clone_schema(
            _base,
            "CLIProxy" + _name,
            "CLIProxy/video/Grok",
            replace_model_options=(
                model_options(_prefixes, ["grok-imagine-video", "grok-imagine-video-1.5", "grok-imagine-video-1.5-preview"])
                if _prefixes
                else None
            ),
        )
        return _extend(schema) if _extend else schema

    return type(
        name,
        (Redirected, base),
        {
            "define_schema": classmethod(_define),
            "execute": classmethod(_video_execute(base)),
        },
    )


def _video_execute(base):
    async def execute(cls, endpoint, api_key, **kwargs):
        model = kwargs.get("model")
        if isinstance(model, str):
            model = model.strip() or DEFAULT_GROK_VIDEO_MODEL
            if base is nodes_grok.GrokVideoExtendNode:
                model = {"model": model, "duration": kwargs.pop("duration")}
            kwargs["model"] = model
        token = cls._with_connection(endpoint, api_key)
        try:
            return await base.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)

    return execute


CLIGrokVideo = _redirected_video(
    "GrokVideo",
    nodes_grok.GrokVideoNode,
    None,
    _chain(
        _free_text_video_model,
        _grok_video_loop_inputs,
        _price_badge(_GROK_VIDEO_BADGE_EXPR, ["model", "duration", "resolution"], ["image"]),
    ),
)
CLIGrokVideoEdit = _redirected_video(
    "GrokVideoEdit", nodes_grok.GrokVideoEditNode, None, _free_text_video_model
)
CLIGrokVideoReference = _redirected_video("GrokVideoReference", nodes_grok.GrokVideoReferenceNode)
def _drop_extend_price_badge(schema):
    schema.price_badge = None
    return schema


CLIGrokVideoExtend = _redirected_video(
    "GrokVideoExtend", nodes_grok.GrokVideoExtendNode, None, _chain(_free_text_video_model, _drop_extend_price_badge)
)


class CLIMuseImage(Redirected, nodes_meta.MuseImageNode):
    @classmethod
    def define_schema(cls):
        schema = clone_schema(nodes_meta.MuseImageNode, "CLIProxyMuseImage", "CLIProxy/image/Meta")
        schema.display_name = "Muse Image"
        return schema

    @classmethod
    async def execute(cls, endpoint, api_key, **kwargs):
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_meta.MuseImageNode.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)


class CLIMuseImageEdit(Redirected, nodes_meta.MuseImageEditNode):
    @classmethod
    def define_schema(cls):
        schema = clone_schema(nodes_meta.MuseImageEditNode, "CLIProxyMuseImageEdit", "CLIProxy/image/Meta")
        schema.display_name = "Muse Image Edit"
        return schema

    @classmethod
    async def execute(cls, endpoint, api_key, **kwargs):
        token = cls._with_connection(endpoint, api_key)
        try:
            return await nodes_meta.MuseImageEditNode.execute.__func__(cls, **kwargs)
        finally:
            cls._reset_connection(token)


class CLIOpenAIGPTImage(Redirected, nodes_openai.OpenAIGPTImageNodeV2):
    @classmethod
    def define_schema(cls):
        return clone_schema(
            nodes_openai.OpenAIGPTImageNodeV2,
            "CLIProxyOpenAIGPTImage",
            "CLIProxy/image/OpenAI",
            replace_model_options=model_options(
                ("gpt-image-",),
                ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "openai/gpt-image-2.5-sunburst", "gpt-image-2", "gpt-image-1.5"],
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
        schema.display_name = "LLM"
        comfy_io = __import__("comfy_api.latest", fromlist=["IO"]).IO
        for item in schema.inputs:
            if getattr(item, "id", None) == "seed":
                item.tooltip = (
                    "Sent as a seed hint and used to bust the Comfy cache. "
                    "Randomize only picks a new seed so the node re-runs. "
                    "It does not make the wording vary. Temperature does that."
                )
        schema.inputs.extend(
            [
                comfy_io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Sampling temperature. 1.0 matches the Qwen3.8 LM Studio load and varies each run. 0 is greedy.",
                ),
                comfy_io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Nucleus sampling. 0.95 is the usual Qwen setting. 1.0 leaves the full distribution.",
                ),
                comfy_io.Int.Input(
                    "top_k",
                    default=20,
                    min=0,
                    max=200,
                    tooltip="Keep only the top K tokens. 0 omits the field. Splash rejects anything outside 1–32.",
                ),
            ]
        )
        schema.inputs.append(
            comfy_io.String.Input(
                "effort",
                display_name="effort",
                default="",
                tooltip="Reasoning effort sent as reasoning.effort. Leave blank to omit reasoning.",
            )
        )
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
    async def execute(
        cls,
        endpoint,
        api_key,
        model,
        image_1=None,
        image_2=None,
        effort="",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        **kwargs,
    ):
        images = [image for image in (image_1, image_2) if image is not None]
        effort_text = effort.strip() if isinstance(effort, str) else ""
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
            if effort_text:
                model_payload["reasoning_effort"] = effort_text
            if images:
                model_payload["images"] = {
                    f"image_{index}": image for index, image in enumerate(images, start=1)
                }
            return await nodes_openrouter.OpenRouterLLMNode.execute.__func__(
                cls,
                model=model_payload,
                temperature=None if temperature < 0 else temperature,
                top_p=None if top_p >= 1 else top_p,
                top_k=None if not top_k else int(top_k),
                **kwargs,
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
    "CLIProxyMuseImage": CLIMuseImage,
    "CLIProxyMuseImageEdit": CLIMuseImageEdit,
    "CLIProxyOpenAIGPTImage": CLIOpenAIGPTImage,
    "CLIProxyGrokVideo": CLIGrokVideo,
    "CLIProxyGrokVideoEdit": CLIGrokVideoEdit,
    "CLIProxyGrokVideoReference": CLIGrokVideoReference,
    "CLIProxyGrokVideoExtend": CLIGrokVideoExtend,
    "CLIProxyOpenRouterLLM": CLIOpenRouterLLM,
    "CLIProxyJSONField": CLIProxyJSONField,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
