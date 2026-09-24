"""Runtime bridge for the copied Comfy API nodes.

The upstream nodes normally call Comfy's hosted API.  This module keeps their
request/response handling and redirects the copied modules to an arbitrary
OpenAI-compatible endpoint supplied by each node.
"""

from __future__ import annotations

import base64
import copy
import json
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from comfy_api.latest import IO
from comfy_api_nodes.util import ApiEndpoint

DEFAULT_ENDPOINT = "http://winpc-1.tailed34e0.ts.net:8317/v1"
DEFAULT_API_KEY = "comfy"
MODEL_DISCOVERY_URL = "http://winpc-1.tailed34e0.ts.net:8317/v1/models?key=one-shot"
# Method paths that people paste into the node by accident. The copied Comfy
# nodes already append /chat/completions, /responses, or /images/*.
_ENDPOINT_METHOD_SUFFIXES = (
    "/chat/completions",
    "/responses/compact",
    "/responses",
    "/completions",
    "/images/generations",
    "/images/edits",
    "/models",
)

_request_context: ContextVar[tuple[str, str] | None] = ContextVar("cliproxy_request_context", default=None)


@dataclass(frozen=True)
class Connection:
    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = DEFAULT_API_KEY


def connection_inputs() -> list:
    return [
        IO.String.Input(
            "endpoint",
            default=DEFAULT_ENDPOINT,
            tooltip="ClipProxy OpenAI base URL ending in /v1. Do not include /chat/completions or /responses.",
            advanced=True,
        ),
        IO.String.Input(
            "api_key",
            default=DEFAULT_API_KEY,
            tooltip="Bearer API key sent to the custom endpoint.",
            advanced=True,
        ),
    ]


def clone_schema(base: type, node_id: str, category: str, *, replace_model_options: list[str] | None = None):
    schema = copy.deepcopy(base.define_schema())
    schema.node_id = node_id
    schema.category = category
    schema.is_api_node = False
    schema.inputs.extend(connection_inputs())
    if replace_model_options is not None:
        for item in schema.inputs:
            if getattr(item, "id", None) == "model" and hasattr(item, "options"):
                if item.options and hasattr(item.options[0], "key"):
                    # DynamicCombo options carry their model ID in ``key``;
                    # preserve their nested quality/size/image inputs.
                    allowed = set(replace_model_options)
                    item.options = [option for option in item.options if option.key in allowed]
                else:
                    item.options = replace_model_options
    return schema


def discover_models() -> list[str]:
    """Read the model list used for simple model dropdowns, with a safe fallback."""
    try:
        req = urllib.request.Request(MODEL_DISCOVERY_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.load(response)
        return [str(item["id"]) for item in payload.get("data", []) if item.get("id")]
    except Exception:
        return []


def model_options(prefixes: tuple[str, ...], fallback: list[str]) -> list[str]:
    all_ids = discover_models()
    discovered = [m for m in all_ids if m.startswith(prefixes)]
    for mid in all_ids:
        if mid.startswith("openai/gpt-image-"):
            alias = "or/" + mid
            if alias not in discovered:
                discovered.append(alias)
    if not discovered:
        return list(fallback)
    out = list(discovered)
    for mid in fallback:
        if mid not in out and not mid.startswith(prefixes):
            out.append(mid)
    return out

def openai_base_url(base_url: str | None) -> str:
    """Return the OpenAI-compatible /v1 root, even if a method URL was pasted."""
    url = (base_url or DEFAULT_ENDPOINT).strip()
    if not url:
        url = DEFAULT_ENDPOINT
    url = url.rstrip("/")
    lowered = url.lower()
    for suffix in _ENDPOINT_METHOD_SUFFIXES:
        if lowered.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    return url or DEFAULT_ENDPOINT


def _absolute_endpoint(endpoint: ApiEndpoint, base_url: str) -> ApiEndpoint:
    path = endpoint.path
    mappings = (
        "/proxy/xai/v1",
        "/proxy/openai",
        "/proxy/openrouter/api/v1",
    )
    for prefix in mappings:
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    return ApiEndpoint(
        path=openai_base_url(base_url).rstrip("/") + "/" + path.lstrip("/"),
        method=endpoint.method,
        query_params=endpoint.query_params,
        headers={"Authorization": f"Bearer {_request_context.get()[1]}"},
    )


def install_redirect(module) -> None:
    """Redirect sync and polling calls in one copied source module."""
    original_sync = module.sync_op
    original_poll = getattr(module, "poll_op", None)

    async def sync_op(cls, endpoint, **kwargs):
        ctx = _request_context.get()
        if ctx:
            endpoint = _absolute_endpoint(endpoint, ctx[0])
        try:
            return await original_sync(cls, endpoint, **kwargs)
        except Exception as exc:
            raise _clarify_request_error(exc, endpoint) from exc

    async def poll_op(cls, endpoint, **kwargs):
        ctx = _request_context.get()
        if ctx:
            endpoint = _absolute_endpoint(endpoint, ctx[0])
            cancel = kwargs.get("cancel_endpoint")
            if cancel is not None:
                kwargs["cancel_endpoint"] = _absolute_endpoint(cancel, ctx[0])
        return await original_poll(cls, endpoint, **kwargs)

    async def inline_images(cls, images, **kwargs):
        return [f"data:image/png;base64,{tensor_to_base64(i)}" for i in images]

    async def inline_video(cls, video, **kwargs):
        stream = video.get_stream_source()
        if hasattr(stream, "read"):
            raw = stream.read()
        else:
            from pathlib import Path as _Path
            path = _Path(stream)
            if not path.is_file():
                raise ValueError(f"Video source is not a readable file: {stream!r}")
            raw = path.read_bytes()
        return f"data:video/mp4;base64,{base64.b64encode(raw).decode()}"

    module.sync_op = sync_op
    if original_poll is not None:
        module.poll_op = poll_op
    module.upload_images_to_comfyapi = inline_images
    module.upload_video_to_comfyapi = inline_video


def tensor_to_base64(tensor) -> str:
    from PIL import Image
    import numpy as np

    value = tensor.detach().cpu()
    if value.ndim == 4:
        value = value[0]
    image = Image.fromarray((value.numpy().clip(0, 1) * 255).astype(np.uint8))
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _clarify_request_error(exc: Exception, endpoint: ApiEndpoint) -> Exception:
    message = str(exc).strip()
    url = getattr(endpoint, "path", "")
    method = getattr(endpoint, "method", "POST")
    if message in ("API Error (raw):", "API Error (raw): ") or message.endswith("(raw):"):
        return Exception(
            f"API Error: empty response from {method} {url}. "
            "Use the ClipProxy base URL ending in /v1, not /chat/completions or /responses."
        )
    if "Unexpected endpoint or method" in message:
        return Exception(
            f"{message} Request went to {method} {url}. "
            "The selected model is routed to a provider whose base-url is missing /v1."
        )
    return exc


class Redirected:
    """Mixin factory used to keep copied node classes small and consistent."""

    @classmethod
    def _with_connection(cls, endpoint: str, api_key: str):
        return _request_context.set((openai_base_url(endpoint), api_key or DEFAULT_API_KEY))

    @classmethod
    def _reset_connection(cls, token):
        _request_context.reset(token)
