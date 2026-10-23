"""Schema checks for the CLIProxy Muse Image nodes."""

from comfy_api_nodes.apis.meta import MuseImageEditRequest

import CLIProxy
from CLIProxy.runtime import _absolute_endpoint, _request_context
from comfy_api_nodes.util import ApiEndpoint


def _ids(schema):
    return [item.id for item in schema.inputs]


def test_muse_image_schema_matches_grok_connection_and_muse_controls():
    schema = CLIProxy.CLIMuseImage.define_schema()
    assert schema.display_name == "Muse Image"
    assert schema.category == "CLIProxy/image/Meta"
    assert schema.node_id == "CLIProxyMuseImage"
    assert _ids(schema) == [
        "model",
        "prompt",
        "aspect_ratio",
        "reasoning_strength",
        "enable_web_search",
        "enable_image_search",
        "enable_shell",
        "seed",
        "endpoint",
        "api_key",
    ]
    model = next(item for item in schema.inputs if item.id == "model")
    assert model.default == "muse-image-1.0"


def test_muse_image_edit_schema_has_reference_images():
    schema = CLIProxy.CLIMuseImageEdit.define_schema()
    assert schema.display_name == "Muse Image Edit"
    assert schema.category == "CLIProxy/image/Meta"
    assert "images" in _ids(schema)
    assert _ids(schema)[-2:] == ["endpoint", "api_key"]


def test_meta_image_path_uses_cliproxy_base():
    token = _request_context.set(("http://winpc-1.tailed34e0.ts.net:8317/v1", "comfy"))
    try:
        _check_meta_paths()
    finally:
        _request_context.reset(token)


def _check_meta_paths():
    endpoint = _absolute_endpoint(
        ApiEndpoint(path="/proxy/meta/v1/images/generations", method="POST"),
        "http://winpc-1.tailed34e0.ts.net:8317/v1",
    )
    assert endpoint.path == "http://winpc-1.tailed34e0.ts.net:8317/v1/images/generations"
    edit = _absolute_endpoint(
        ApiEndpoint(path="/proxy/meta/v1/images/edits", method="POST"),
        "http://winpc-1.tailed34e0.ts.net:8317/v1/images/generations",
    )
    assert edit.path == "http://winpc-1.tailed34e0.ts.net:8317/v1/images/edits"


def test_edit_request_keeps_openai_image_url_shape():
    body = MuseImageEditRequest(
        model="muse-image-1.0",
        prompt="A red fox",
        images=[{"image_url": "data:image/png;base64,abc"}],
    ).model_dump(exclude_none=True)
    assert body["model"] == "muse-image-1.0"
    assert body["images"] == [{"image_url": "data:image/png;base64,abc"}]
    assert body["response_format"] == "b64_json"
