"""CLIProxy Muse Image nodes.

Same controls as the partner Meta Muse image nodes, sent to the local
CLIProxy OpenAI base instead of Comfy's hosted Meta proxy.
"""

import base64
import re
from io import BytesIO

import torch

from comfy_api.latest import IO
from comfy_api_nodes.apis.meta import (
    MuseImageEditRequest,
    MuseImageInput,
    MuseImageRequest,
    MuseImageResponse,
    MuseImageToolEnablement,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_to_image_tensor,
    pad_images_to_common_channels,
    sync_op,
    tensor_to_base64_string,
    validate_string,
)

GENERATIONS_PATH = "/proxy/meta/v1/images/generations"
EDITS_PATH = "/proxy/meta/v1/images/edits"
MUSE_IMAGE_MODEL = "muse-image-1.0"
MAX_INPUT_IMAGES = 10
ASPECT_RATIOS = ["auto", "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16", "21:9", "9:21", "2:1", "1:2"]
REASONING_STRENGTHS = ["high", "low"]

_IMAGE_REF_RE = re.compile(r"@image(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)


def _resolve_image_refs(prompt: str, total_images: int) -> str:
    parts = []
    pos = 0
    prev_end = -1
    for match in _IMAGE_REF_RE.finditer(prompt):
        start = match.start()
        if start > 0 and start != prev_end and (prompt[start - 1].isalnum() or prompt[start - 1] == "_"):
            continue
        idx = int(match.group("idx") or 1)
        if not 1 <= idx <= total_images:
            raise ValueError(
                f"The prompt references @Image{idx}, but only {total_images} reference images "
                f"are connected (a batched input counts once per image)."
            )
        parts.append(prompt[pos:start])
        parts.append(f"image {idx}")
        pos = match.end()
        prev_end = match.end()
    parts.append(prompt[pos:])
    return "".join(parts)


def _size(aspect_ratio: str) -> str | None:
    return None if aspect_ratio == "auto" else aspect_ratio.replace(":", "x")


async def _decode_images(cls: type[IO.ComfyNode], response: MuseImageResponse) -> torch.Tensor:
    images = []
    for item in response.data:
        if item.b64_json:
            images.append(bytesio_to_image_tensor(BytesIO(base64.b64decode(item.b64_json))))
        elif item.url:
            images.append(await download_url_to_image_tensor(item.url, cls=cls))
    if not images:
        raise Exception("The response contains no images.")
    return torch.cat(pad_images_to_common_channels(images))


def _prompt_input(tooltip: str) -> IO.String.Input:
    return IO.String.Input("prompt", multiline=True, default="", tooltip=tooltip)


def _aspect_ratio_input(tooltip: str) -> IO.Combo.Input:
    return IO.Combo.Input("aspect_ratio", options=ASPECT_RATIOS, tooltip=tooltip)


def _reasoning_strength_input() -> IO.Combo.Input:
    return IO.Combo.Input(
        "reasoning_strength",
        options=REASONING_STRENGTHS,
        tooltip="How much the model thinks, plans and self-refines before rendering.",
    )


def _tool_toggle_inputs() -> list[IO.Boolean.Input]:
    return [
        IO.Boolean.Input(
            "enable_web_search",
            default=True,
            tooltip="Lets the model search the web for facts and live information while planning the image.",
        ),
        IO.Boolean.Input(
            "enable_image_search",
            default=True,
            tooltip="Lets the model search for reference images while planning the image.",
        ),
        IO.Boolean.Input(
            "enable_shell",
            default=True,
            tooltip="Lets the model run code while planning, for precise layouts, charts and diagrams; "
            "when off, quantities and alignment are approximated.",
        ),
    ]


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        step=1,
        display_mode=IO.NumberDisplay.number,
        control_after_generate=True,
        tooltip="Seed to determine if node should re-run; the API has no seed, "
        "so actual results are nondeterministic regardless of this value.",
    )


def _images_input() -> IO.Autogrow.Input:
    return IO.Autogrow.Input(
        "images",
        template=IO.Autogrow.TemplateNames(
            IO.Image.Input("image"),
            names=[f"image_{i}" for i in range(1, MAX_INPUT_IMAGES + 1)],
            min=1,
        ),
        tooltip=f"1-{MAX_INPUT_IMAGES} reference images to edit or combine. Refer to them in the prompt "
        "as @Image1, @Image2, ..., numbered in input order; a batched input counts once per image.",
    )


def _tool_enablement(
    enable_web_search: bool, enable_image_search: bool, enable_shell: bool
) -> MuseImageToolEnablement | None:
    if enable_web_search and enable_image_search and enable_shell:
        return None
    return MuseImageToolEnablement(
        enable_image_search=enable_image_search,
        enable_web_search=enable_web_search,
        enable_shell=enable_shell,
    )


def _flatten_images(images) -> list:
    if not images:
        return []
    if isinstance(images, dict):
        ordered = []
        for key in images:
            value = images[key]
            if value is None:
                continue
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                ordered.extend(value[i] for i in range(value.shape[0]))
            else:
                ordered.append(value)
        return ordered
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        return [images[i] for i in range(images.shape[0])]
    return [images]


class MuseImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CLIProxyMuseImage",
            display_name="Muse Image",
            category="CLIProxy/image/Meta",
            description="Generate an image from a text prompt with Meta Muse Image through CLIProxy.",
            inputs=[
                IO.String.Input(
                    "model",
                    default=MUSE_IMAGE_MODEL,
                    tooltip="CLIProxy image model ID, e.g. muse-image-1.0 or meta/muse-image-1.0.",
                ),
                _prompt_input(
                    "Prompt describing the image. The model reasons about the prompt, and may use "
                    "its built-in web and image search, before rendering."
                ),
                _aspect_ratio_input(
                    "Aspect ratio of the output. Images are rendered at about 2.5 megapixels "
                    "(1:1 is 1600x1600, 16:9 is 2048x1152); 'auto' lets the model choose from the prompt."
                ),
                _reasoning_strength_input(),
                *_tool_toggle_inputs(),
                _seed_input(),
            ],
            outputs=[IO.Image.Output()],
            hidden=[IO.Hidden.unique_id],
            is_api_node=False,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str,
        reasoning_strength: str,
        enable_web_search: bool,
        enable_image_search: bool,
        enable_shell: bool,
        seed: int,
    ):
        del seed
        validate_string(prompt, min_length=1)
        response = await sync_op(
            cls,
            ApiEndpoint(path=GENERATIONS_PATH, method="POST"),
            response_model=MuseImageResponse,
            data=MuseImageRequest(
                model=(model or "").strip() or MUSE_IMAGE_MODEL,
                prompt=prompt,
                size=_size(aspect_ratio),
                reasoning_strength=reasoning_strength,
                tool_enablement=_tool_enablement(enable_web_search, enable_image_search, enable_shell),
            ),
        )
        return IO.NodeOutput(await _decode_images(cls, response))


class MuseImageEditNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CLIProxyMuseImageEdit",
            display_name="Muse Image Edit",
            category="CLIProxy/image/Meta",
            description=f"Edit or combine up to {MAX_INPUT_IMAGES} reference images with Meta Muse Image through CLIProxy.",
            inputs=[
                IO.String.Input(
                    "model",
                    default=MUSE_IMAGE_MODEL,
                    tooltip="CLIProxy image model ID, e.g. muse-image-1.0 or meta/muse-image-1.0.",
                ),
                _images_input(),
                _prompt_input("Editing instructions. Supports @Image1-style references to the input images."),
                _aspect_ratio_input(
                    "Aspect ratio of the output. Images are rendered at about 2.5 megapixels "
                    "(1:1 is 1600x1600, 16:9 is 2048x1152); 'auto' keeps the aspect ratio of the input."
                ),
                _reasoning_strength_input(),
                *_tool_toggle_inputs(),
                _seed_input(),
            ],
            outputs=[IO.Image.Output()],
            hidden=[IO.Hidden.unique_id],
            is_api_node=False,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str,
        reasoning_strength: str,
        enable_web_search: bool,
        enable_image_search: bool,
        enable_shell: bool,
        seed: int,
        images=None,
    ):
        del seed
        validate_string(prompt, min_length=1)
        reference_images = _flatten_images(images)
        if not reference_images:
            raise ValueError("At least one image is required for editing.")
        if len(reference_images) > MAX_INPUT_IMAGES:
            raise ValueError(
                f"A maximum of {MAX_INPUT_IMAGES} reference images is supported; got {len(reference_images)} "
                "(a batched input counts once per image)."
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path=EDITS_PATH, method="POST"),
            response_model=MuseImageResponse,
            data=MuseImageEditRequest(
                model=(model or "").strip() or MUSE_IMAGE_MODEL,
                prompt=_resolve_image_refs(prompt, len(reference_images)),
                size=_size(aspect_ratio),
                reasoning_strength=reasoning_strength,
                tool_enablement=_tool_enablement(enable_web_search, enable_image_search, enable_shell),
                images=[
                    MuseImageInput(image_url=f"data:image/png;base64,{tensor_to_base64_string(image, total_pixels=None)}")
                    for image in reference_images
                ],
            ),
        )
        return IO.NodeOutput(await _decode_images(cls, response))
