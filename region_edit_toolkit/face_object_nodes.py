from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.special import ndtr


MAX_RESOLUTION = 16384
MULTIPLE = 16
MIN_CONTEXT_RETENTION_RATIO = 0.90
SOURCE_EDIT_CONTEXT_POLICY = "源图头部编辑（严格上下文）"
SOURCE_EDGE_COMPAT_CONTEXT_POLICY = "源图边缘兼容（裁边不中止）"
IDENTITY_REFERENCE_CONTEXT_POLICY = "身份参考脸（完整五官优先）"
CONTEXT_POLICIES = (
    SOURCE_EDIT_CONTEXT_POLICY,
    SOURCE_EDGE_COMPAT_CONTEXT_POLICY,
    IDENTITY_REFERENCE_CONTEXT_POLICY,
)

BROAD_HEAD_REGION_MODE = "宽松头部语义区（推荐）"
LEGACY_FACE_OVAL_MODE = "旧版脸椭圆外扩（兼容）"
EDIT_REGION_MODES = (BROAD_HEAD_REGION_MODE, LEGACY_FACE_OVAL_MODE)

# The broad mode is deliberately a semantic editing envelope, not an attempt to
# trace individual hair strands.  These ratios place the editable boundary
# outside the facial oval so the model can reconstruct bangs, hairline, ears and
# upper neck coherently.  Manual add/erase/protection remains authoritative.
BROAD_HEAD_SIDE_RATIO = 0.50
BROAD_HEAD_TOP_RATIO = 0.75
BROAD_HEAD_BOTTOM_RATIO = 0.20
BROAD_NECK_BOTTOM_RATIO = 0.55

# The mandatory core covers the head, face and a short upper-neck transition.
# It does not force the shoulders or chest to generated pixels.  Those outer
# regions may still enter the final writeback through the independent
# source/generated difference transition, so a real model change is never
# truncated merely because it lies outside this geometric core.
BROAD_COMPOSITE_HEAD_BOTTOM_RATIO = 0.08
BROAD_COMPOSITE_NECK_BOTTOM_RATIO = 0.18
BROAD_COMPOSITE_NECK_HALF_TOP_RATIO = 0.32
BROAD_COMPOSITE_NECK_HALF_BOTTOM_RATIO = 0.34

# Adaptive strict-writeback v4 uses a compact mandatory head/face core and
# reserves the source/generated difference field for the surrounding seam.
# All distances scale from the detected face, so the same contract applies to
# close portraits and small faces inside a whole-person crop.
ADAPTIVE_CORE_GUARD_FACE_RATIO = 0.006
ADAPTIVE_COMPATIBILITY_BAND_FACE_RATIO = 0.060
ADAPTIVE_DIFFERENCE_FULL_STRENGTH_FACE_RATIO = 0.220
ADAPTIVE_DIFFERENCE_OUTER_LIMIT_FACE_RATIO = 0.300
ADAPTIVE_BOUNDARY_SUPPRESSION_FACE_RATIO = 0.060

# API runners send this exact contract string to the semantic crop node.  The
# value is also exposed through /object_info as the optional input default, so
# a runner can refuse to submit when ComfyUI still has an older copy of this
# node loaded in memory after an on-disk update.
WRITEBACK_SCOPE_CONTRACT_VERSION = "phase1-head-upper-neck-v1"

# A separate runtime handshake protects the color-continuity path from the
# same disk-new / memory-old failure mode as the writeback planner.  This node
# computes statistics from valid context pixels only; excluded generation
# pixels are never zero-padded into the mean or standard deviation.
COLOR_HARMONIZATION_CONTRACT_VERSION = "face-local-exact-valid-context-lab-v1"
COLOR_HARMONIZATION_MINIMUM_CONTEXT_PIXELS = 4096

# Source iris/pupil/catchlight preservation is deliberately isolated from both
# identity-reference conditioning and the broad face writeback mask.  It is a
# small semantic material restore performed inside the generated local crop,
# after color harmonization and before the existing strict full-frame merge.
SOURCE_EYE_MATERIAL_RESTORE_CONTRACT_VERSION = "face-local-source-gaze-aligned-iris-material-v3"

FACE_TILE_PROFILES = {
    "标准（单脸）": (1536, 1024, 1_572_864),
    "大块（高显存）": (2048, 1536, 3_145_728),
    "超大块（RTX 4090）": (2816, 2816, 8_388_608),
}

STRUCTURE_GROUPS = {
    "eyes": "visibly change the eye shape and eye spacing",
    "nose": "visibly change the nose bridge and nose tip structure",
    "mouth": "visibly change the mouth shape and lip proportions",
    "jaw": "visibly change the jawline and overall facial contour",
    "brows": "visibly change the eyebrow shape and brow-to-eye relationship",
}


ROUTE_MODES = ("N1", "R1", "R2", "N2")
DEFAULT_N1_PROMPT_EN = (
    "Regenerate the woman's face as a natural East Asian woman with a clearly different facial identity. "
    "Keep the source head pose, gaze, expression, hairstyle, facial lighting, body, clothing, composition, "
    "and background."
)
DEFAULT_N1_PROMPT_ZH = (
    "将源图中女性的面部重新生成为自然的东亚女性，并形成明显不同的面部身份。"
    "保持源图的头部姿态、视线、表情、发型、面部光照、身体、服装、构图和背景。"
)
DEFAULT_R1_PROMPT_EN = (
    "Transfer the facial identity from Image 2 to the woman in Image 1, using Image 2 for the brows, nose, "
    "mouth, jaw, and overall feature combination. Keep Image 1's face shape, irises, pupils, eye catchlights, "
    "gaze, expression, facial lighting, head pose, hairstyle, body, clothing, composition, and background."
)
DEFAULT_R1_PROMPT_ZH = (
    "将图像 2 的面部身份迁移给图像 1 中的女性，并使用图像 2 的眉形、鼻部、嘴部、下颌和整体五官组合。"
    "保持图像 1 的脸型、虹膜、瞳孔、眼部高光、视线、表情、面部光照、头部姿态、发型、身体、服装、构图和背景。"
)
DEFAULT_R2_N2_PROMPT_EN = (
    "Replace the facial identity of the woman in Image 1 with the facial identity of the woman in Image 2. "
    "Keep Image 1's head pose, gaze, expression, hairstyle, lighting, body, clothing, composition, and background."
)
DEFAULT_R2_N2_PROMPT_ZH = (
    "将图像 1 中女性的面部身份替换为图像 2 中女性的面部身份。"
    "保持图像 1 的头部姿态、视线、表情、发型、光照、身体、服装、构图和背景。"
)

EYE_IDENTITY_EXCLUSION_MODES = ("none", "iris_only", "visible_eye_interior")
IRIS_LANDMARK_GROUPS = (
    (468, (469, 470, 471, 472)),
    (473, (474, 475, 476, 477)),
)
VISIBLE_EYE_CONTOURS = (
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
)
IRIS_GAZE_FRAME_GROUPS = (
    (468, 33, 133, VISIBLE_EYE_CONTOURS[0]),
    (473, 362, 263, VISIBLE_EYE_CONTOURS[1]),
)


def _image_batch(image: torch.Tensor) -> torch.Tensor:
    """Copied into this project from the verified native tile implementation."""
    value = image.detach().to(dtype=torch.float32)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"IMAGE must be BHWC, received shape {tuple(value.shape)}")
    if value.shape[0] != 1:
        raise ValueError("Phase one accepts exactly one input image at a time")
    return value[..., :3]


def _plan_exact_integer_downscale(
    width: int,
    height: int,
    maximum_short_edge: int,
    latent_multiple: int = MULTIPLE,
) -> tuple[int, int, int]:
    """Choose the smallest exact integer divisor that satisfies a short-edge cap.

    Both model dimensions remain divisible by ``latent_multiple`` and multiplying
    them by the returned divisor restores the input dimensions exactly.  This
    deliberately rejects arbitrary total-pixel resizing because that can make a
    VAE silently trim unmatched edge pixels.
    """
    width = int(width)
    height = int(height)
    maximum_short_edge = int(maximum_short_edge)
    latent_multiple = int(latent_multiple)

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if latent_multiple <= 0:
        raise ValueError("latent_multiple must be positive")
    if maximum_short_edge < latent_multiple or maximum_short_edge % latent_multiple != 0:
        raise ValueError(
            f"maximum_short_edge must be a positive multiple of {latent_multiple}"
        )
    if width % latent_multiple != 0 or height % latent_multiple != 0:
        raise ValueError(
            f"Input {width}x{height} is not aligned to {latent_multiple}; "
            "align the upstream crop instead of silently trimming model pixels"
        )

    width_units = width // latent_multiple
    height_units = height // latent_multiple
    common_units = math.gcd(width_units, height_units)
    divisors = []
    for candidate in range(1, math.isqrt(common_units) + 1):
        if common_units % candidate == 0:
            divisors.append(candidate)
            paired = common_units // candidate
            if paired != candidate:
                divisors.append(paired)

    for divisor in sorted(divisors):
        model_width = width // divisor
        model_height = height // divisor
        if min(model_width, model_height) <= maximum_short_edge:
            if model_width % latent_multiple != 0 or model_height % latent_multiple != 0:
                raise AssertionError("Integer scale planner produced an unaligned model canvas")
            return model_width, model_height, divisor

    raise ValueError(
        f"Input {width}x{height} has no exact integer downscale whose short edge is at most "
        f"{maximum_short_edge} and whose dimensions remain multiples of {latent_multiple}. "
        f"Align the upstream crop to {latent_multiple} x the required integer divisor."
    )


def _resize_lanczos_exact(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    source = _image_batch(image)
    source_height, source_width = int(source.shape[1]), int(source.shape[2])
    if (source_width, source_height) == (int(width), int(height)):
        return source

    samples = source.movedim(-1, 1)
    try:
        import comfy.utils  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name != "comfy" and not str(exc.name).startswith("comfy."):
            raise
        # Keeps the node unit-testable outside ComfyUI.  The installed runtime
        # always takes the native Lanczos path above.
        resized = torch.nn.functional.interpolate(
            samples,
            size=(int(height), int(width)),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    else:
        resized = comfy.utils.common_upscale(
            samples,
            int(width),
            int(height),
            "lanczos",
            "disabled",
        )
    result = resized.movedim(1, -1)
    if tuple(result.shape[1:3]) != (int(height), int(width)):
        raise RuntimeError(
            f"Exact resize returned {result.shape[2]}x{result.shape[1]}, "
            f"expected {width}x{height}"
        )
    return result


def _mask_batch(mask: torch.Tensor | np.ndarray, height: int, width: int) -> torch.Tensor:
    """Copied into this project from the verified native tile implementation."""
    if isinstance(mask, torch.Tensor):
        value = mask.detach().to(device="cpu", dtype=torch.float32)
    else:
        value = torch.from_numpy(np.asarray(mask, dtype=np.float32))
    if value.ndim == 4:
        if value.shape[-1] == 1:
            value = value[..., 0]
        elif value.shape[1] == 1:
            value = value[:, 0]
        else:
            value = value.amax(dim=-1)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"MASK must be BHW, received shape {tuple(value.shape)}")
    if value.shape[0] != 1:
        value = value.amax(dim=0, keepdim=True)
    if value.shape[-2:] != (height, width):
        value = torch.nn.functional.interpolate(
            value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
    return value.clamp(0.0, 1.0)


def _crop_intersection(
    x: int,
    y: int,
    local_width: int,
    local_height: int,
    original_width: int,
    original_height: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Map a possibly padded local crop to its valid original-image intersection.

    Whole-person crops deliberately include context and may therefore begin at
    a negative coordinate or extend past the source canvas.  The padded pixels
    remain useful model context, but only the intersecting source pixels are
    eligible for semantic-mask lookup or strict writeback.
    """
    dimensions = (
        int(local_width),
        int(local_height),
        int(original_width),
        int(original_height),
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("Crop and original image dimensions must be positive")

    original_x0 = max(0, int(x))
    original_y0 = max(0, int(y))
    original_x1 = min(int(original_width), int(x) + int(local_width))
    original_y1 = min(int(original_height), int(y) + int(local_height))
    if original_x1 <= original_x0 or original_y1 <= original_y0:
        raise ValueError("Planned crop does not intersect the original image")

    local_x0 = original_x0 - int(x)
    local_y0 = original_y0 - int(y)
    local_x1 = local_x0 + (original_x1 - original_x0)
    local_y1 = local_y0 + (original_y1 - original_y0)
    return (
        original_x0,
        original_y0,
        original_x1,
        original_y1,
        local_x0,
        local_y0,
        local_x1,
        local_y1,
    )


def _srgb_to_lab(image_bhwc: torch.Tensor) -> torch.Tensor:
    """Convert clamped sRGB BHWC tensors to CIE Lab BCHW (D65) in pure torch."""
    rgb = image_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
    )
    red, green, blue = linear.unbind(dim=1)
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883
    delta = 6.0 / 29.0
    threshold = delta**3

    def pivot(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            value > threshold,
            torch.pow(value.clamp_min(0.0), 1.0 / 3.0),
            value / (3.0 * delta**2) + 4.0 / 29.0,
        )

    fx, fy, fz = pivot(x), pivot(y), pivot(z)
    return torch.stack(
        (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)),
        dim=1,
    )


def _lab_to_srgb(lab_bchw: torch.Tensor) -> torch.Tensor:
    """Convert CIE Lab BCHW (D65) to sRGB BHWC in pure torch."""
    lightness, channel_a, channel_b = lab_bchw.unbind(dim=1)
    fy = (lightness + 16.0) / 116.0
    fx = fy + channel_a / 500.0
    fz = fy - channel_b / 200.0
    delta = 6.0 / 29.0

    def inverse_pivot(value: torch.Tensor) -> torch.Tensor:
        return torch.where(
            value > delta,
            value**3,
            3.0 * delta**2 * (value - 4.0 / 29.0),
        )

    x = 0.95047 * inverse_pivot(fx)
    y = inverse_pivot(fy)
    z = 1.08883 * inverse_pivot(fz)
    red = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    green = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    blue = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    linear = torch.stack((red, green, blue), dim=1)
    srgb = torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * torch.pow(linear.clamp_min(0.0), 1.0 / 2.4) - 0.055,
    )
    return srgb.permute(0, 2, 3, 1)


def _exact_masked_channel_stats(
    values_bchw: torch.Tensor,
    valid_context_bhw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return population std/mean/count using only true context pixels."""
    weights = valid_context_bhw.to(device=values_bchw.device, dtype=values_bchw.dtype).unsqueeze(1)
    count = weights.sum(dim=(2, 3), keepdim=True)
    safe_count = count.clamp_min(1.0)
    mean = (values_bchw * weights).sum(dim=(2, 3), keepdim=True) / safe_count
    variance = (((values_bchw - mean) ** 2) * weights).sum(dim=(2, 3), keepdim=True) / safe_count
    return variance.clamp_min(0.0).sqrt(), mean, count


def _expand_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Euclidean expansion copied from the verified strict-merge implementation."""
    support = np.asarray(mask, dtype=bool)
    if pixels <= 0 or not np.any(support):
        return support.copy()
    ys, xs = np.nonzero(support)
    height, width = support.shape
    y0 = max(0, int(ys.min()) - pixels)
    y1 = min(height, int(ys.max()) + 1 + pixels)
    x0 = max(0, int(xs.min()) - pixels)
    x1 = min(width, int(xs.max()) + 1 + pixels)
    roi = support[y0:y1, x0:x1]
    expanded_roi = ndimage.distance_transform_edt(~roi) <= float(pixels)
    expanded = np.zeros_like(support)
    expanded[y0:y1, x0:x1] = expanded_roi
    return expanded


def _inward_feather_alpha(
    support: np.ndarray,
    blur_map: np.ndarray | int | float,
) -> np.ndarray:
    """Verified global inward feather, with exact zero outside the support.

    This is the single-tile form of a global-alpha implementation. A scalar is
    accepted for compatibility, while a per-pixel map keeps generation growth
    and final feathering independent.
    Source-image edges are open boundaries because no compositing seam exists
    beyond the image.
    """
    hard = np.asarray(support, dtype=bool)
    alpha = np.zeros(hard.shape, dtype=np.float32)
    if not np.any(hard):
        return alpha
    if np.isscalar(blur_map):
        feather = float(blur_map)
        local_blur_map = np.full(hard.shape, feather, dtype=np.float32)
    else:
        local_blur_map = np.asarray(blur_map, dtype=np.float32)
        if local_blur_map.shape != hard.shape:
            raise ValueError(
                f"blur_map shape {local_blur_map.shape} does not match support {hard.shape}"
            )
    ys, xs = np.nonzero(hard)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    roi = hard[y0:y1, x0:x1]
    padded = np.pad(roi, 1, mode="constant", constant_values=False)
    # A support region may deliberately run to a source-image edge so hair is not
    # cut by an artificial line inside the image.  In that case there is no
    # compositing boundary outside the source, so do not fade alpha there.
    if y0 == 0:
        padded[0, 1:-1] = roi[0, :]
    if y1 == hard.shape[0]:
        padded[-1, 1:-1] = roi[-1, :]
    if x0 == 0:
        padded[1:-1, 0] = roi[:, 0]
    if x1 == hard.shape[1]:
        padded[1:-1, -1] = roi[:, -1]
    if y0 == 0 and x0 == 0:
        padded[0, 0] = roi[0, 0]
    if y0 == 0 and x1 == hard.shape[1]:
        padded[0, -1] = roi[0, -1]
    if y1 == hard.shape[0] and x0 == 0:
        padded[-1, 0] = roi[-1, 0]
    if y1 == hard.shape[0] and x1 == hard.shape[1]:
        padded[-1, -1] = roi[-1, -1]
    distance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    local_blur = local_blur_map[y0:y1, x0:x1]
    local = np.ones(roi.shape, dtype=np.float32)
    feathered = roi & (local_blur > 0)
    t = np.zeros(roi.shape, dtype=np.float32)
    t[feathered] = np.clip(
        distance[feathered] / local_blur[feathered],
        0.0,
        1.0,
    )
    local[feathered] = t[feathered] * t[feathered] * (3.0 - 2.0 * t[feathered])
    local[~roi] = 0.0
    alpha[y0:y1, x0:x1] = local.astype(np.float32, copy=False)
    return alpha


def _broad_head_semantic_seed(
    target: np.ndarray,
    face_bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Build a smooth head/hair/ear/upper-neck envelope around the face.

    The face oval and any manual additions remain part of the semantic request.
    The geometric envelope merely prevents the final boundary from crossing the
    forehead or bangs; it intentionally does not pretend to be a strand-exact
    hair matte.
    """
    support = np.asarray(target, dtype=bool)
    height, width = support.shape
    x1, y1, x2, y2 = (float(v) for v in face_bbox)
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) * 0.5

    semantic_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(semantic_image)
    draw.ellipse(
        (
            x1 - face_width * BROAD_HEAD_SIDE_RATIO,
            y1 - face_height * BROAD_HEAD_TOP_RATIO,
            x2 + face_width * BROAD_HEAD_SIDE_RATIO,
            y2 + face_height * BROAD_HEAD_BOTTOM_RATIO,
        ),
        fill=255,
    )
    neck_top = y2 - face_height * 0.08
    neck_bottom = y2 + face_height * BROAD_NECK_BOTTOM_RATIO
    draw.polygon(
        (
            (center_x - face_width * 0.34, neck_top),
            (center_x + face_width * 0.34, neck_top),
            (center_x + face_width * 0.43, neck_bottom),
            (center_x - face_width * 0.43, neck_bottom),
        ),
        fill=255,
    )
    envelope = np.asarray(semantic_image, dtype=np.uint8) > 0
    return support | envelope


def _broad_head_composite_seed(
    target: np.ndarray,
    face_bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Build the independent strict-writeback envelope.

    ``target`` is expected to contain a mature semantic face+hair mask.  The
    geometric envelope adds ears, jaw/chin and a short upper-neck transition.
    This core itself does not force body pixels, but the separate adaptive
    difference transition may append coherent changes anywhere inside the
    whole-person processing support.
    """
    support = np.asarray(target, dtype=bool)
    height, width = support.shape
    x1, y1, x2, y2 = (float(v) for v in face_bbox)
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) * 0.5

    semantic_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(semantic_image)
    draw.ellipse(
        (
            x1 - face_width * BROAD_HEAD_SIDE_RATIO,
            y1 - face_height * BROAD_HEAD_TOP_RATIO,
            x2 + face_width * BROAD_HEAD_SIDE_RATIO,
            y2 + face_height * BROAD_COMPOSITE_HEAD_BOTTOM_RATIO,
        ),
        fill=255,
    )
    neck_top = y2 - face_height * 0.06
    neck_bottom = y2 + face_height * BROAD_COMPOSITE_NECK_BOTTOM_RATIO
    draw.polygon(
        (
            (
                center_x - face_width * BROAD_COMPOSITE_NECK_HALF_TOP_RATIO,
                neck_top,
            ),
            (
                center_x + face_width * BROAD_COMPOSITE_NECK_HALF_TOP_RATIO,
                neck_top,
            ),
            (
                center_x + face_width * BROAD_COMPOSITE_NECK_HALF_BOTTOM_RATIO,
                neck_bottom,
            ),
            (
                center_x - face_width * BROAD_COMPOSITE_NECK_HALF_BOTTOM_RATIO,
                neck_bottom,
            ),
        ),
        fill=255,
    )
    envelope = np.asarray(semantic_image, dtype=np.uint8) > 0
    return support | envelope


def _disk_structure(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _compact_complete_face_core(
    shape: tuple[int, int],
    face_bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a compact forehead-to-chin guard for the detected face.

    The semantic mask remains authoritative for hairstyle/head coverage.  This
    compact convex guard only closes detector/segmenter holes across the
    forehead, cheeks, jaw and chin; unlike the legacy broad ellipse it does not
    force a large rectangular or shoulder-shaped generated patch.
    """
    height, width = (int(shape[0]), int(shape[1]))
    x1, y1, x2, y2 = (float(value) for value in face_bbox)
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    normalized_points = (
        (0.10, 0.00),
        (0.90, 0.00),
        (0.98, 0.20),
        (1.00, 0.54),
        (0.88, 0.79),
        (0.66, 1.00),
        (0.34, 1.00),
        (0.12, 0.79),
        (0.00, 0.54),
        (0.02, 0.20),
    )
    points = [
        (
            min(width - 1, max(0, int(round(x1 + nx * face_width)))),
            min(height - 1, max(0, int(round(y1 + ny * face_height)))),
        )
        for nx, ny in normalized_points
    ]
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon(points, fill=255)
    mask = np.asarray(image, dtype=np.uint8) > 0
    guard_radius = max(1, int(round(min(face_width, face_height) * 0.012)))
    return ndimage.binary_dilation(
        mask,
        structure=_disk_structure(guard_radius),
    )


def _smoothstep_numpy(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if not edge1 > edge0:
        raise ValueError("edge1 must be greater than edge0")
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _ordered_rings(edges: Any) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {}
    for raw_a, raw_b in edges:
        a, b = int(raw_a), int(raw_b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    visited: set[int] = set()
    rings: list[list[int]] = []
    for start in adjacency:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        previous, current = -1, start
        while True:
            candidates = sorted(v for v in adjacency[current] if v != previous)
            next_value = next((v for v in candidates if v == start or v not in visited), None)
            if next_value is None or next_value == start:
                break
            ring.append(next_value)
            visited.add(next_value)
            previous, current = current, next_value
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _face_candidate_metrics(
    face: dict[str, Any],
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    bbox = np.asarray(face.get("bbox_xyxy", ()), dtype=np.float64)
    landmarks = np.asarray(face.get("landmarks_xy", ()), dtype=np.float64)
    bbox_valid = bool(
        bbox.shape == (4,)
        and np.isfinite(bbox).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )
    area = (
        float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if bbox_valid
        else 0.0
    )
    confidence = float(face.get("score", 0.0))
    if not math.isfinite(confidence):
        confidence = 0.0

    landmarks_complete = bool(
        landmarks.ndim == 2
        and landmarks.shape[0] >= 468
        and landmarks.shape[1] >= 2
        and np.isfinite(landmarks[:, :2]).all()
    )
    landmark_inside_bbox_fraction = 0.0
    if bbox_valid and landmarks_complete:
        xy = landmarks[:, :2]
        inside = (
            (xy[:, 0] >= bbox[0])
            & (xy[:, 0] <= bbox[2])
            & (xy[:, 1] >= bbox[1])
            & (xy[:, 1] <= bbox[3])
        )
        landmark_inside_bbox_fraction = float(np.mean(inside))

    bbox_inside_image_fraction = 0.0
    if bbox_valid and image_size is not None:
        image_height, image_width = (int(v) for v in image_size)
        ix1 = max(0.0, float(bbox[0]))
        iy1 = max(0.0, float(bbox[1]))
        ix2 = min(float(image_width), float(bbox[2]))
        iy2 = min(float(image_height), float(bbox[3]))
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        bbox_inside_image_fraction = intersection / area if area > 0.0 else 0.0

    candidate_valid = bool(
        bbox_valid
        and landmarks_complete
        and (image_size is None or bbox_inside_image_fraction >= 0.98)
    )
    return {
        "confidence": confidence,
        "bbox_area": area,
        "bbox_valid": bbox_valid,
        "landmarks_complete": landmarks_complete,
        "landmark_inside_bbox_fraction": landmark_inside_bbox_fraction,
        "bbox_inside_image_fraction": bbox_inside_image_fraction,
        "candidate_valid": candidate_valid,
    }


def _detected_faces(face_landmarks: dict[str, Any]) -> list[dict[str, Any]]:
    frames = face_landmarks.get("frames") or []
    if not frames or not frames[0]:
        return []
    image_size = face_landmarks.get("image_size")
    faces = []
    for original_index, raw_face in enumerate(frames[0]):
        face = dict(raw_face)
        face["_face_local_original_detection_index"] = int(original_index)
        face["_face_local_candidate_metrics"] = _face_candidate_metrics(face, image_size)
        faces.append(face)
    return sorted(
        faces,
        key=lambda face: (
            -int(face["_face_local_candidate_metrics"]["candidate_valid"]),
            -float(face["_face_local_candidate_metrics"]["confidence"]),
            -float(
                face["_face_local_candidate_metrics"]["landmark_inside_bbox_fraction"]
            ),
            -float(face["_face_local_candidate_metrics"]["bbox_inside_image_fraction"]),
            -float(face["_face_local_candidate_metrics"]["bbox_area"]),
        ),
    )


def _sorted_faces(face_landmarks: dict[str, Any]) -> list[dict[str, Any]]:
    faces = _detected_faces(face_landmarks)
    if not faces:
        raise ValueError("No face was detected; generation is blocked")
    return faces


def _selection_face(selection: dict[str, Any]) -> dict[str, Any]:
    face = selection.get("face")
    if face is None:
        raise ValueError("Invalid FACE_LOCAL_SELECTION")
    return face


def _aligned_axis(
    start: float,
    stop: float,
    limit: int,
    multiple: int = MULTIPLE,
    *,
    allow_source_edge_trim: bool = False,
) -> tuple[int, int]:
    required_start = max(0, int(math.floor(start)))
    required_stop = min(limit, int(math.ceil(stop)))
    if required_stop <= required_start:
        raise ValueError("Required crop axis is empty")
    aligned_start = int(math.floor(required_start / multiple) * multiple)
    aligned_stop = int(math.ceil(required_stop / multiple) * multiple)
    if aligned_stop > limit:
        length = int(math.ceil((required_stop - required_start) / multiple) * multiple)
        while limit - length > required_start:
            length += multiple
        if length > limit and allow_source_edge_trim:
            # Odd-sized sources may be smaller than the outward-rounded latent
            # extent by at most multiple-1 pixels. Under the explicit
            # source-edge-compatible policy these border pixels are context,
            # so keep the largest aligned crop inside the source.
            length = (limit // multiple) * multiple
            if length < multiple:
                raise ValueError("Input dimension is smaller than one aligned semantic crop")
            if required_start == 0:
                return 0, length
            return limit - length, limit
        if length > limit:
            raise ValueError("Input dimension is smaller than one aligned semantic crop")
        aligned_start = limit - length
        aligned_stop = limit
    if aligned_start > required_start or aligned_stop < required_stop:
        raise RuntimeError("Aligned crop failed to contain the required semantic region")
    if (aligned_stop - aligned_start) % multiple:
        raise RuntimeError("Aligned crop length is not a multiple of the required latent alignment")
    return aligned_start, aligned_stop


def _overlay_mask(image: torch.Tensor, mask: torch.Tensor, color: tuple[float, float, float]) -> torch.Tensor:
    source = _image_batch(image).to(device="cpu")
    height, width = int(source.shape[1]), int(source.shape[2])
    alpha = _mask_batch(mask, height, width).unsqueeze(-1) * 0.48
    tint = torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand_as(source)
    return (source * (1.0 - alpha) + tint * alpha).clamp(0.0, 1.0)


class FaceLocalSelectFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_landmarks": ("FACE_LANDMARKS",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "minimum_face_short_side": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 16},
                ),
            }
        }

    RETURN_TYPES = ("FACE_LOCAL_SELECTION", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("selection", "automatic_face_mask", "selection_preview", "report_json")
    FUNCTION = "select"
    CATEGORY = "face local edit/1 locate"

    def select(self, image, face_landmarks, face_index, minimum_face_short_side):
        source = _image_batch(image).to(device="cpu")
        height, width = int(source.shape[1]), int(source.shape[2])
        faces = _sorted_faces(face_landmarks)
        index = int(face_index)
        if index >= len(faces):
            raise ValueError(f"face_index {index} is out of range for {len(faces)} detected face(s)")
        face = faces[index]
        x1, y1, x2, y2 = (float(v) for v in face["bbox_xyxy"])
        face_width, face_height = x2 - x1, y2 - y1
        face_short = min(face_width, face_height)
        if face_short < int(minimum_face_short_side):
            raise ValueError(
                f"Selected face short side is {face_short:.1f}px, below the hard gate "
                f"of {int(minimum_face_short_side)}px"
            )

        rings = _ordered_rings(face_landmarks["connection_sets"]["face_oval"])
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        landmarks = np.asarray(face["landmarks_xy"], dtype=np.float32)
        for ring in rings:
            draw.polygon(
                [(float(landmarks[i, 0]), float(landmarks[i, 1])) for i in ring], fill=255
            )
        mask_np = np.asarray(mask_image, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask_np.copy()).unsqueeze(0)

        preview_np = (source[0].numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        preview_image = Image.fromarray(preview_np)
        preview_draw = ImageDraw.Draw(preview_image)
        for number, detected in enumerate(faces):
            bx1, by1, bx2, by2 = (float(v) for v in detected["bbox_xyxy"])
            color = (0, 255, 80) if number == index else (255, 190, 0)
            preview_draw.rectangle((bx1, by1, bx2, by2), outline=color, width=4)
            preview_draw.text((bx1 + 6, by1 + 6), str(number), fill=color)
        preview = torch.from_numpy(np.asarray(preview_image).copy()).float().div(255.0).unsqueeze(0)
        selection = {
            "face": face,
            "face_index": index,
            "detected_count": len(faces),
            "image_width": width,
            "image_height": height,
        }
        candidate_rankings = []
        for rank, detected in enumerate(faces):
            metrics = dict(detected["_face_local_candidate_metrics"])
            candidate_rankings.append(
                {
                    "rank": rank,
                    "original_detection_index": int(
                        detected["_face_local_original_detection_index"]
                    ),
                    "bbox_xyxy": [float(v) for v in detected["bbox_xyxy"]],
                    **metrics,
                }
            )
        report = {
            "detected_face_count": len(faces),
            "selected_face_index_ranked": index,
            "selected_original_detection_index": int(
                face["_face_local_original_detection_index"]
            ),
            "ranking_policy": (
                "valid_bbox_and_complete_landmarks_then_detection_confidence_then_"
                "landmark_geometry_then_area"
            ),
            "candidate_rankings": candidate_rankings,
            "bbox_xyxy": [x1, y1, x2, y2],
            "face_width": face_width,
            "face_height": face_height,
            "face_short_side": face_short,
            "minimum_face_short_side": int(minimum_face_short_side),
            "gate_passed": True,
        }
        return selection, mask, preview, json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalFaceContextSquareCrop:
    """Crop adaptive face context while treating ``crop_size`` as a hard maximum.

    The crop grows from the detected face so a distant subject does not turn a
    1024 px limit into a forced 1024 x 1024 full-body crop. It keeps generous
    hair, shoulder, neck, and upper-torso context, but never enlarges the source
    image and never exceeds the configured maximum side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "selection": ("FACE_LOCAL_SELECTION",),
                "crop_size": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 64},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("face_context_crop", "report_json")
    FUNCTION = "crop"
    CATEGORY = "face local edit/2 reference condition"

    def crop(self, image, selection, crop_size):
        source = _image_batch(image).to(device="cpu")
        height, width = int(source.shape[1]), int(source.shape[2])
        requested_maximum = int(crop_size)
        if requested_maximum <= 0:
            raise ValueError("crop_size must be positive")

        expected_width = int(selection.get("image_width", width))
        expected_height = int(selection.get("image_height", height))
        if (expected_width, expected_height) != (width, height):
            raise ValueError(
                "Face selection dimensions do not match the image: "
                f"selection={expected_width}x{expected_height}, image={width}x{height}"
            )

        face = selection.get("face")
        if not isinstance(face, dict) or "bbox_xyxy" not in face:
            raise ValueError("FACE_LOCAL_SELECTION does not contain a valid face bounding box")
        x1, y1, x2, y2 = (float(value) for value in face["bbox_xyxy"])
        face_width = x2 - x1
        face_height = y2 - y1
        if face_width <= 0.0 or face_height <= 0.0:
            raise ValueError("Selected face bounding box is empty")

        # ``crop_size`` is a ceiling, not a target. Four face widths and five
        # face heights retain hair plus useful shoulder/upper-torso context.
        # The 256 px floor avoids an unusably tiny identity reference for a
        # distant face, while the source short side prevents any crop upscaling.
        source_aligned_maximum = int(min(width, height, requested_maximum) // 16 * 16)
        if source_aligned_maximum < 16:
            raise ValueError("The source image is too small for a 16-aligned face crop")
        desired_context = max(face_width * 4.0, face_height * 5.0, 256.0)
        desired_aligned = int(math.ceil(desired_context / 16.0) * 16)
        size = min(source_aligned_maximum, desired_aligned)

        face_center_x = (x1 + x2) * 0.5
        # The base envelope reserves 0.25 face-heights above the detected box
        # and 0.75 below it. Extra room is split 35/65, favouring upper torso.
        base_top = y1 - face_height * 0.25
        base_height = face_height * 2.0
        extra_height = max(0.0, float(size) - base_height)
        desired_top = base_top - extra_height * 0.35
        desired_left = face_center_x - float(size) * 0.5

        left = min(max(int(round(desired_left)), 0), max(0, width - size))
        top = min(max(int(round(desired_top)), 0), max(0, height - size))

        right = left + size
        bottom = top + size
        source_left = max(0, left)
        source_top = max(0, top)
        source_right = min(width, right)
        source_bottom = min(height, bottom)
        if source_right <= source_left or source_bottom <= source_top:
            raise ValueError("Requested face-context crop does not overlap the image")

        crop = source[:, source_top:source_bottom, source_left:source_right, :]
        pad_left = source_left - left
        pad_top = source_top - top
        pad_right = right - source_right
        pad_bottom = bottom - source_bottom
        if any(value > 0 for value in (pad_left, pad_right, pad_top, pad_bottom)):
            crop = torch.nn.functional.pad(
                crop.permute(0, 3, 1, 2),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            ).permute(0, 2, 3, 1)
        if tuple(crop.shape[1:3]) != (size, size):
            raise RuntimeError(
                f"Face-context crop must be exactly {size}x{size}, got "
                f"{int(crop.shape[2])}x{int(crop.shape[1])}"
            )

        report = {
            "configured_maximum_crop_size": requested_maximum,
            "adaptive_crop_size": size,
            "crop_xyxy": [left, top, right, bottom],
            "source_intersection_xyxy": [
                source_left,
                source_top,
                source_right,
                source_bottom,
            ],
            "padding_left_top_right_bottom": [
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            "face_bbox_xyxy": [x1, y1, x2, y2],
            "face_width": face_width,
            "face_height": face_height,
            "face_width_ratio_in_crop": face_width / float(size),
            "face_height_ratio_in_crop": face_height / float(size),
            "face_vertical_position_ratio": ((y1 + y2) * 0.5 - top) / float(size),
            "context_policy": "adaptive_face_relative_square_with_hard_maximum",
            "output_width": size,
            "output_height": size,
        }
        return crop.contiguous(), json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalIdentityConditioningMask:
    """Remove selected eye material from identity-reference conditioning only.

    This mask never participates in the final output composite.  Its purpose is
    to stop a reference identity from injecting iris/pupil material while the
    source image remains available as reference index 0.  The output boundary
    therefore cannot become a pasted-eye seam.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_landmarks": ("FACE_LANDMARKS",),
                "base_mask": ("MASK",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "eye_exclusion_mode": (list(EYE_IDENTITY_EXCLUSION_MODES), {"default": "none"}),
                "eye_exclusion_scale": (
                    "FLOAT",
                    {"default": 1.8, "min": 0.75, "max": 3.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "identity_conditioning_mask",
        "excluded_eye_material_mask",
        "preview",
        "report_json",
    )
    FUNCTION = "build"
    CATEGORY = "face local edit/2 reference condition"

    def build(
        self,
        image,
        face_landmarks,
        base_mask,
        face_index,
        eye_exclusion_mode,
        eye_exclusion_scale,
    ):
        source = _image_batch(image).to(device="cpu")
        height, width = int(source.shape[1]), int(source.shape[2])
        base = _mask_batch(base_mask, height, width)
        mode = str(eye_exclusion_mode)
        if mode not in EYE_IDENTITY_EXCLUSION_MODES:
            raise ValueError(f"Unsupported eye identity exclusion mode: {mode}")
        scale = float(eye_exclusion_scale)
        if not math.isfinite(scale) or not 0.75 <= scale <= 3.0:
            raise ValueError("eye_exclusion_scale must be finite and between 0.75 and 3.0")

        faces = _sorted_faces(face_landmarks)
        index = int(face_index)
        if index >= len(faces):
            raise ValueError(f"face_index {index} is out of range for {len(faces)} detected face(s)")
        points = np.asarray(faces[index]["landmarks_xy"], dtype=np.float32)
        required_points = 478 if mode == "iris_only" else 467
        if mode != "none" and points.shape[0] < required_points:
            raise ValueError(
                f"{mode} requires at least {required_points} MediaPipe landmarks; "
                f"received {points.shape[0]}"
            )
        if mode != "none" and not np.all(np.isfinite(points[:required_points])):
            raise ValueError("Face landmarks contain non-finite coordinates")

        exclusion_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(exclusion_image)
        regions: list[dict[str, Any]] = []
        if mode == "iris_only":
            for center_index, boundary_indices in IRIS_LANDMARK_GROUPS:
                center = points[center_index]
                boundary = points[list(boundary_indices)]
                radius_x = max(1.0, float(np.max(np.abs(boundary[:, 0] - center[0]))) * scale)
                radius_y = max(1.0, float(np.max(np.abs(boundary[:, 1] - center[1]))) * scale)
                if radius_x <= 1.0 and radius_y <= 1.0:
                    raise ValueError("Iris landmarks collapsed; identity conditioning is blocked")
                box = (
                    float(center[0] - radius_x),
                    float(center[1] - radius_y),
                    float(center[0] + radius_x),
                    float(center[1] + radius_y),
                )
                draw.ellipse(box, fill=255)
                regions.append(
                    {
                        "center_landmark": center_index,
                        "center_xy": [float(center[0]), float(center[1])],
                        "radius_xy": [radius_x, radius_y],
                    }
                )
        elif mode == "visible_eye_interior":
            for contour_indices in VISIBLE_EYE_CONTOURS:
                contour = points[list(contour_indices)]
                center = contour.mean(axis=0)
                scaled = center + (contour - center) * scale
                draw.polygon([(float(x), float(y)) for x, y in scaled], fill=255)
                regions.append(
                    {
                        "contour_landmarks": list(contour_indices),
                        "center_xy": [float(center[0]), float(center[1])],
                    }
                )

        exclusion_np = np.asarray(exclusion_image, dtype=np.float32) / 255.0
        exclusion = torch.from_numpy(exclusion_np.copy()).unsqueeze(0)
        conditioning = (base * (1.0 - exclusion)).clamp(0.0, 1.0)
        base_support = base > 0.001
        excluded_inside = base_support & (exclusion > 0.001)
        final_support = conditioning > 0.001
        if mode != "none" and int(torch.count_nonzero(excluded_inside)) == 0:
            raise ValueError("Eye exclusion does not overlap the identity conditioning mask")
        if int(torch.count_nonzero(final_support)) == 0:
            raise ValueError("Eye exclusion removed the entire identity conditioning mask")

        preview = _overlay_mask(source, base, (0.0, 0.8, 0.15))
        preview = _overlay_mask(preview, exclusion, (1.0, 0.0, 0.0))
        report = {
            "eye_exclusion_mode": mode,
            "eye_exclusion_scale": scale,
            "selected_face_index_largest_first": index,
            "regions": regions,
            "base_conditioning_pixels": int(torch.count_nonzero(base_support)),
            "excluded_eye_pixels_inside_base": int(torch.count_nonzero(excluded_inside)),
            "final_conditioning_pixels": int(torch.count_nonzero(final_support)),
            "affects_identity_reference_conditioning_only": True,
            "affects_generation_mask": False,
            "affects_final_composite_mask": False,
            "creates_output_pixel_boundary": False,
            "gate_passed": True,
        }
        return conditioning, exclusion, preview, json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalSourceEyeMaterialRestore:
    """Restore source iris material at the source gaze inside generated eyes.

    The source iris position is measured in an eye-local frame formed by the
    source eye corners and aperture, then mapped into the corresponding generated
    eye frame. This preserves source gaze without pasting at stale full-frame
    coordinates or accepting the generated model's divergent pupil position.
    Eyelids, lashes, brows and under-eye structure remain generated. The local
    generation mask remains authoritative, and pixels outside the aligned restore
    support are copied byte-for-byte from ``generated_local``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_local": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "local_generation_mask": ("MASK",),
                "source_landmarks": ("FACE_LANDMARKS",),
                "generated_landmarks": ("FACE_LANDMARKS",),
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "generated_face_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "crop_x": ("INT", {"forceInput": True}),
                "crop_y": ("INT", {"forceInput": True}),
                "iris_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.65, "max": 1.5, "step": 0.05},
                ),
                "feather_pixels": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 32.0, "step": 0.5},
                ),
            },
            "optional": {
                "source_eye_material_restore_contract_version": (
                    "STRING",
                    {"default": SOURCE_EYE_MATERIAL_RESTORE_CONTRACT_VERSION},
                )
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "restored_local",
        "iris_restore_alpha",
        "preview",
        "report_json",
    )
    FUNCTION = "restore"
    CATEGORY = "face local edit/4 continuity"

    def restore(
        self,
        source_local,
        generated_local,
        local_generation_mask,
        source_landmarks,
        generated_landmarks,
        source_face_index,
        generated_face_index,
        crop_x,
        crop_y,
        iris_scale,
        feather_pixels,
        source_eye_material_restore_contract_version=SOURCE_EYE_MATERIAL_RESTORE_CONTRACT_VERSION,
    ):
        loaded_contract = str(source_eye_material_restore_contract_version)
        if loaded_contract != SOURCE_EYE_MATERIAL_RESTORE_CONTRACT_VERSION:
            raise ValueError(
                "FaceLocalSourceEyeMaterialRestore runtime contract mismatch: "
                f"expected {SOURCE_EYE_MATERIAL_RESTORE_CONTRACT_VERSION!r}, "
                f"received {loaded_contract!r}"
            )
        source = _image_batch(source_local).to(device="cpu", dtype=torch.float32)
        generated = _image_batch(generated_local).to(device="cpu", dtype=torch.float32)
        if source.shape != generated.shape:
            raise ValueError(
                f"source_local shape {tuple(source.shape)} does not match "
                f"generated_local shape {tuple(generated.shape)}"
            )
        height, width = int(source.shape[1]), int(source.shape[2])
        generation = _mask_batch(local_generation_mask, height, width)
        scale = float(iris_scale)
        feather = float(feather_pixels)
        if not math.isfinite(scale) or not 0.65 <= scale <= 1.5:
            raise ValueError("iris_scale must be finite and between 0.65 and 1.5")
        if not math.isfinite(feather) or not 0.0 <= feather <= 32.0:
            raise ValueError("feather_pixels must be finite and between 0 and 32")

        source_faces = _sorted_faces(source_landmarks)
        source_index = int(source_face_index)
        if source_index >= len(source_faces):
            raise ValueError(
                f"source_face_index {source_index} is out of range for "
                f"{len(source_faces)} detected face(s)"
            )
        generated_faces = _sorted_faces(generated_landmarks)
        generated_index = int(generated_face_index)
        if generated_index >= len(generated_faces):
            raise ValueError(
                f"generated_face_index {generated_index} is out of range for "
                f"{len(generated_faces)} detected face(s)"
            )
        source_points = np.asarray(
            source_faces[source_index]["landmarks_xy"], dtype=np.float32
        )
        generated_points = np.asarray(
            generated_faces[generated_index]["landmarks_xy"], dtype=np.float32
        )
        if source_points.shape[0] < 478 or generated_points.shape[0] < 478:
            raise ValueError(
                "aligned source iris material restore requires at least 478 MediaPipe "
                f"landmarks for source and generated faces; received "
                f"{source_points.shape[0]} and {generated_points.shape[0]}"
            )
        if not np.all(np.isfinite(source_points[:478])):
            raise ValueError("Source face landmarks contain non-finite coordinates")
        if not np.all(np.isfinite(generated_points[:478])):
            raise ValueError("Generated face landmarks contain non-finite coordinates")

        offset_x, offset_y = float(crop_x), float(crop_y)
        hard_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(hard_image)
        regions: list[dict[str, Any]] = []
        aligned_source = generated.clone()
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        gaze_frames = {
            center_index: (corner_a, corner_b, contour)
            for center_index, corner_a, corner_b, contour in IRIS_GAZE_FRAME_GROUPS
        }
        for center_index, boundary_indices in IRIS_LANDMARK_GROUPS:
            source_center_full = source_points[center_index]
            source_boundary = source_points[list(boundary_indices)]
            source_radius_x = float(
                np.max(np.abs(source_boundary[:, 0] - source_center_full[0]))
            )
            source_radius_y = float(
                np.max(np.abs(source_boundary[:, 1] - source_center_full[1]))
            )
            generated_center_local = generated_points[center_index]
            generated_boundary = generated_points[list(boundary_indices)]
            generated_radius_x = float(
                np.max(np.abs(generated_boundary[:, 0] - generated_center_local[0]))
            ) * scale
            generated_radius_y = float(
                np.max(np.abs(generated_boundary[:, 1] - generated_center_local[1]))
            ) * scale

            corner_a_index, corner_b_index, contour_indices = gaze_frames[center_index]
            source_corner_a = source_points[corner_a_index]
            source_corner_b = source_points[corner_b_index]
            generated_corner_a = generated_points[corner_a_index]
            generated_corner_b = generated_points[corner_b_index]
            source_eye_vector = source_corner_b - source_corner_a
            generated_eye_vector = generated_corner_b - generated_corner_a
            source_eye_width = float(np.linalg.norm(source_eye_vector))
            generated_eye_width = float(np.linalg.norm(generated_eye_vector))
            if min(source_eye_width, generated_eye_width) <= 2.0:
                raise ValueError("Eye-corner landmarks collapsed; source-gaze restore is blocked")
            source_eye_unit = source_eye_vector / source_eye_width
            generated_eye_unit = generated_eye_vector / generated_eye_width
            source_eye_perp = np.asarray(
                [-source_eye_unit[1], source_eye_unit[0]], dtype=np.float32
            )
            generated_eye_perp = np.asarray(
                [-generated_eye_unit[1], generated_eye_unit[0]], dtype=np.float32
            )
            source_eye_midpoint = (source_corner_a + source_corner_b) * 0.5
            generated_eye_midpoint = (generated_corner_a + generated_corner_b) * 0.5
            source_contour = source_points[list(contour_indices)]
            generated_contour = generated_points[list(contour_indices)]
            source_half_aperture = float(
                np.max(np.abs((source_contour - source_eye_midpoint) @ source_eye_perp))
            )
            generated_half_aperture = float(
                np.max(
                    np.abs(
                        (generated_contour - generated_eye_midpoint) @ generated_eye_perp
                    )
                )
            )
            if min(source_half_aperture, generated_half_aperture) <= 1.0:
                raise ValueError("Eye-aperture landmarks collapsed; source-gaze restore is blocked")
            source_center_delta = source_center_full - source_eye_midpoint
            source_gaze_u = float(
                np.dot(source_center_delta, source_eye_vector)
                / (source_eye_width * source_eye_width)
            )
            source_gaze_v = float(
                np.dot(source_center_delta, source_eye_perp) / source_half_aperture
            )
            if not math.isfinite(source_gaze_u) or not math.isfinite(source_gaze_v):
                raise ValueError("Source gaze coordinates are non-finite")
            if not -0.35 <= source_gaze_u <= 0.35 or not -2.5 <= source_gaze_v <= 2.5:
                raise ValueError(
                    "Source iris falls outside the supported eye-local gaze frame; restore is blocked"
                )
            if min(
                source_radius_x,
                source_radius_y,
                generated_radius_x,
                generated_radius_y,
            ) <= 1.0:
                raise ValueError("Iris landmarks collapsed; aligned eye restore is blocked")
            source_center_local = (
                float(source_center_full[0] - offset_x),
                float(source_center_full[1] - offset_y),
            )
            target_center_array = (
                generated_eye_midpoint
                + source_gaze_u * generated_eye_vector
                + source_gaze_v * generated_half_aperture * generated_eye_perp
            )
            target_center = (float(target_center_array[0]), float(target_center_array[1]))
            box = (
                target_center[0] - generated_radius_x,
                target_center[1] - generated_radius_y,
                target_center[0] + generated_radius_x,
                target_center[1] + generated_radius_y,
            )
            draw.ellipse(box, fill=255)

            # Map every target-iris coordinate back into the corresponding
            # source iris ellipse. Only pixels inside the target ellipse are
            # later used, so the full-frame sampling grid cannot alter context.
            source_x = source_center_local[0] + (
                (xx - target_center[0]) / generated_radius_x
            ) * source_radius_x
            source_y = source_center_local[1] + (
                (yy - target_center[1]) / generated_radius_y
            ) * source_radius_y
            grid_x = source_x.mul(2.0 / max(width - 1, 1)).sub(1.0)
            grid_y = source_y.mul(2.0 / max(height - 1, 1)).sub(1.0)
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
            sampled = torch.nn.functional.grid_sample(
                source.permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            ellipse = (
                ((xx - target_center[0]) / generated_radius_x).square()
                + ((yy - target_center[1]) / generated_radius_y).square()
                <= 1.0
            )
            aligned_source[:, ellipse, :] = sampled[:, ellipse, :]
            regions.append(
                {
                    "center_landmark": center_index,
                    "source_center_full_xy": [
                        float(source_center_full[0]),
                        float(source_center_full[1]),
                    ],
                    "source_center_local_xy": list(source_center_local),
                    "source_radius_xy": [source_radius_x, source_radius_y],
                    "source_gaze_eye_frame_uv": [source_gaze_u, source_gaze_v],
                    "generated_iris_landmark_center_local_xy": [
                        float(generated_center_local[0]),
                        float(generated_center_local[1]),
                    ],
                    "source_gaze_mapped_center_local_xy": list(target_center),
                    "generated_radius_xy": [generated_radius_x, generated_radius_y],
                    "source_center_to_mapped_center_shift_xy": [
                        target_center[0] - source_center_local[0],
                        target_center[1] - source_center_local[1],
                    ],
                    "generated_landmark_vs_mapped_center_delta_xy": [
                        float(generated_center_local[0]) - target_center[0],
                        float(generated_center_local[1]) - target_center[1],
                    ],
                }
            )

        hard = np.asarray(hard_image, dtype=np.uint8) > 0
        generation_support = generation[0].numpy() > 0.5
        hard &= generation_support
        if not np.any(hard):
            raise ValueError("Source iris restore support does not overlap the local generation mask")
        alpha_np = _inward_feather_alpha(hard, feather)
        alpha = torch.from_numpy(alpha_np.copy()).unsqueeze(0)
        alpha_bhwc = alpha.unsqueeze(-1)
        restored = generated.clone()
        blended = generated * (1.0 - alpha_bhwc) + aligned_source * alpha_bhwc
        support = alpha > 0.0
        restored[support] = blended[support]
        outside_exact = torch.equal(restored[~support], generated[~support])
        if not outside_exact:
            raise RuntimeError("Source eye material restore changed pixels outside its alpha support")

        preview = _overlay_mask(restored, alpha, (0.0, 1.0, 1.0))
        fully_restored = alpha >= (1.0 - 1.0e-6)
        report = {
            "source_eye_material_restore_contract_version": (
                SOURCE_EYE_MATERIAL_RESTORE_CONTRACT_VERSION
            ),
            "operation": "source-gaze-aligned-iris-pupil-catchlight-material-restore",
            "source_face_index_largest_first": source_index,
            "generated_face_index_largest_first": generated_index,
            "crop_origin_xy": [int(crop_x), int(crop_y)],
            "local_width": width,
            "local_height": height,
            "iris_scale": scale,
            "feather_pixels": feather,
            "regions": regions,
            "restore_support_pixels": int(torch.count_nonzero(support)),
            "fully_restored_source_pixels": int(torch.count_nonzero(fully_restored)),
            "outside_restore_support_is_exact_generated": outside_exact,
            "limited_by_local_generation_mask": True,
            "source_material_preserved": ["iris", "pupil", "catchlight"],
            "source_material_alignment": (
                "source iris material mapped to source gaze position in generated eye frame"
            ),
            "generated_regions_left_editable": [
                "eye_shape",
                "eyelids",
                "eyelashes",
                "brows",
                "under_eye_detail",
            ],
            "does_not_change_full_frame_writeback_support": True,
            "visual_eye_alignment_and_seam_review_still_required": True,
            "gate_passed": bool(outside_exact and int(torch.count_nonzero(fully_restored)) > 0),
        }
        return restored.clamp(0.0, 1.0), alpha, preview, json.dumps(
            report, ensure_ascii=False, indent=2
        )


class FaceLocalComposeMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "automatic_mask": ("MASK",),
                "manual_add_mask": ("MASK",),
                "manual_erase_mask": ("MASK",),
                "protection_mask": ("MASK",),
                "support_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("final_target_mask", "confirmed_protection_mask", "mask_preview", "report_json")
    FUNCTION = "compose"
    CATEGORY = "face local edit/2 mask"

    def compose(
        self,
        image,
        automatic_mask,
        manual_add_mask,
        manual_erase_mask,
        protection_mask,
        support_threshold,
    ):
        source = _image_batch(image).to(device="cpu")
        height, width = int(source.shape[1]), int(source.shape[2])
        automatic = _mask_batch(automatic_mask, height, width)
        add = _mask_batch(manual_add_mask, height, width)
        erase = _mask_batch(manual_erase_mask, height, width)
        protection = _mask_batch(protection_mask, height, width)
        # In broad head-region mode, erase is not merely a hole in the anatomical
        # seed: it is an explicit request to keep those pixels.  Return it with the
        # protection mask so the later broad regeneration window cannot cover it
        # again accidentally.
        confirmed_protection = torch.maximum(protection, erase)
        target = torch.maximum(automatic, add)
        target = torch.clamp(target - confirmed_protection, 0.0, 1.0)
        target[target < float(support_threshold)] = 0.0
        if not torch.any(target > 0):
            raise ValueError("The final face mask is empty after erase/protection subtraction")

        overlay = _overlay_mask(source, target, (1.0, 0.05, 0.05))
        protect_alpha = confirmed_protection.unsqueeze(-1) * 0.42
        blue = torch.tensor((0.05, 0.25, 1.0), dtype=torch.float32).view(1, 1, 1, 3)
        overlay = overlay * (1.0 - protect_alpha) + blue * protect_alpha
        report = {
            "automatic_pixels": int(torch.count_nonzero(automatic > 0).item()),
            "manual_add_pixels": int(torch.count_nonzero(add > 0).item()),
            "manual_erase_pixels": int(torch.count_nonzero(erase > 0).item()),
            "protection_pixels": int(torch.count_nonzero(protection > 0).item()),
            "confirmed_keep_original_pixels": int(
                torch.count_nonzero(confirmed_protection > 0).item()
            ),
            "final_target_pixels": int(torch.count_nonzero(target > 0).item()),
            "formula": "(automatic union add) minus (erase union protection)",
            "manual_erase_becomes_keep_original": True,
            "gate_passed": True,
        }
        return (
            target,
            confirmed_protection,
            overlay.clamp(0.0, 1.0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalMaskEditorCanvas:
    """Prepare an internal red/cyan MaskEditor canvas.

    Connect ``mask_editor_rgba`` directly to Impact Pack ``PreviewBridge``.
    The bridge exposes ComfyUI's MaskEditor without requiring a third opening
    ``LoadImage`` input.  Once the canvas has been saved by MaskEditor, its MASK
    output is the complete user-edited mask because ComfyUI defines it as
    ``1 - PNG alpha``.  Impact Pack returns a 64x64 empty placeholder before
    that first save; the correction node detects that placeholder and keeps the
    automatic mask instead of treating it as an intentional full erase.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "automatic_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "mask_editor_rgba",
        "red_cyan_visible_preview",
        "automatic_mask_passthrough",
        "report_json",
    )
    FUNCTION = "prepare"
    CATEGORY = "face local edit/2 mask"

    def prepare(self, source_image, generated_image, automatic_mask):
        source = _image_batch(source_image).to(device="cpu")
        generated = _image_batch(generated_image).to(device="cpu")
        if source.shape != generated.shape:
            raise ValueError(
                "source_image and generated_image must have identical dimensions"
            )
        height, width = int(source.shape[1]), int(source.shape[2])
        automatic = _mask_batch(automatic_mask, height, width)

        luminance_weights = torch.tensor(
            (0.299, 0.587, 0.114), dtype=torch.float32
        ).view(1, 1, 1, 3)
        source_gray = torch.sum(source * luminance_weights, dim=-1, keepdim=True)
        generated_gray = torch.sum(
            generated * luminance_weights, dim=-1, keepdim=True
        )
        red_cyan = torch.cat(
            (source_gray, generated_gray, generated_gray), dim=-1
        ).clamp(0.0, 1.0)
        inverse_alpha = (1.0 - automatic).unsqueeze(-1)
        rgba = torch.cat((red_cyan, inverse_alpha), dim=-1).clamp(0.0, 1.0)

        report = {
            "operation": "prepare-red-cyan-maskeditor-canvas",
            "image_size": [width, height],
            "automatic_support_pixels": int(
                torch.count_nonzero(automatic > (0.5 / 255.0)).item()
            ),
            "rgba_alpha_contract": "ComfyUI LoadImage MASK = 1 - PNG alpha",
            "interaction": [
                "send mask_editor_rgba directly to Impact Pack PreviewBridge",
                "open the PreviewBridge image in core MaskEditor",
                "paint to add and erase to remove transition support",
                "connect the PreviewBridge MASK to the manual correction node",
            ],
            "external_image_input_required": False,
            "gate_passed": True,
        }
        return (
            rgba,
            red_cyan,
            automatic,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalManualAdaptiveMaskCorrection:
    """Apply a complete MaskEditor edit without allowing face-core erasure.

    ``PreviewBridge`` returns a 64x64 empty placeholder until MaskEditor has
    saved a canvas.  A placeholder therefore must not replace an existing
    automatic mask.  A same-size empty mask is different: it is a legitimate
    user edit that erased every non-mandatory pixel.

    ``apply_manual_editor`` remains as a compatibility input for older
    workflows.  New workflows keep it enabled internally and do not expose it
    as a user-facing switch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "difference_image": ("IMAGE",),
                "automatic_mask": ("MASK",),
                "edited_mask": ("MASK",),
                "mandatory_core_mask": ("MASK",),
                "support_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
            "optional": {
                "processing_support_mask": ("MASK",),
                "apply_manual_editor": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "use edited mask",
                        "label_off": "use automatic mask",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "final_mask",
        "manual_add_mask",
        "manual_erase_mask",
        "correction_preview",
        "report_json",
    )
    FUNCTION = "correct"
    CATEGORY = "face local edit/2 mask"

    def correct(
        self,
        difference_image,
        automatic_mask,
        edited_mask,
        mandatory_core_mask,
        support_threshold=0.001,
        processing_support_mask=None,
        apply_manual_editor=True,
    ):
        difference = _image_batch(difference_image).to(device="cpu")
        height, width = int(difference.shape[1]), int(difference.shape[2])
        automatic = _mask_batch(automatic_mask, height, width)
        edited_value = (
            edited_mask.detach().to(device="cpu", dtype=torch.float32)
            if isinstance(edited_mask, torch.Tensor)
            else torch.from_numpy(np.asarray(edited_mask, dtype=np.float32))
        )
        if edited_value.ndim == 4:
            if edited_value.shape[-1] == 1:
                raw_editor_size = tuple(int(v) for v in edited_value.shape[1:3])
            elif edited_value.shape[1] == 1:
                raw_editor_size = tuple(int(v) for v in edited_value.shape[2:4])
            else:
                raw_editor_size = tuple(int(v) for v in edited_value.shape[1:3])
        elif edited_value.ndim in (2, 3):
            raw_editor_size = tuple(int(v) for v in edited_value.shape[-2:])
        else:
            raise ValueError(
                f"MASK must be BHW, received shape {tuple(edited_value.shape)}"
            )
        raw_edited = _mask_batch(edited_mask, height, width)
        manual_editor_requested = bool(apply_manual_editor)
        editor_canvas_initialized = raw_editor_size == (height, width)
        uninitialized_editor_fallback = bool(
            manual_editor_requested and not editor_canvas_initialized
        )
        manual_editor_applied = bool(
            manual_editor_requested and editor_canvas_initialized
        )
        edited = raw_edited if manual_editor_applied else automatic
        core = _mask_batch(mandatory_core_mask, height, width)
        support = (
            torch.ones_like(automatic)
            if processing_support_mask is None
            else _mask_batch(processing_support_mask, height, width)
        )
        threshold = float(support_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        core_selected = core > threshold
        support_selected = support > threshold
        if torch.any(core_selected & ~support_selected):
            raise ValueError("mandatory_core_mask extends outside processing support")

        manual_add = torch.clamp(edited - automatic, 0.0, 1.0)
        manual_erase = torch.clamp(automatic - edited, 0.0, 1.0)
        protected_core_restore = torch.clamp(core - edited, 0.0, 1.0)
        final_mask = torch.maximum(edited, core)
        final_mask = torch.where(support_selected, final_mask, torch.zeros_like(final_mask))
        final_mask[final_mask < threshold] = 0.0
        final_mask[core_selected] = 1.0

        preview = difference.clone()
        layers = (
            (manual_add, (0.05, 1.0, 0.15)),
            (manual_erase, (1.0, 0.05, 0.85)),
            (protected_core_restore, (1.0, 0.90, 0.05)),
        )
        for layer, color_values in layers:
            weight = torch.clamp(layer.unsqueeze(-1) * 0.68, 0.0, 0.68)
            color = torch.tensor(color_values, dtype=torch.float32).view(
                1, 1, 1, 3
            )
            preview = preview * (1.0 - weight) + color * weight

        core_lock_passed = bool(torch.all(final_mask[core_selected] == 1.0))
        outside_support_passed = bool(
            torch.all(final_mask[~support_selected] == 0.0)
        )
        report = {
            "algorithm": "maskeditor-auto-apply-with-uninitialized-placeholder-fallback-and-mandatory-core-lock-v3",
            "formula": (
                "clip_to_processing_support(max(mandatory_core, "
                "edited_complete_mask if editor_canvas_initialized else automatic_mask))"
            ),
            "manual_editor_requested": manual_editor_requested,
            "manual_editor_applied": manual_editor_applied,
            "manual_editor_bypassed": not manual_editor_applied,
            "editor_canvas_initialized": editor_canvas_initialized,
            "uninitialized_editor_fallback": uninitialized_editor_fallback,
            "raw_editor_size_hw": list(raw_editor_size),
            "expected_editor_size_hw": [height, width],
            "raw_editor_support_pixels": int(
                torch.count_nonzero(raw_edited > threshold).item()
            ),
            "automatic_support_pixels": int(
                torch.count_nonzero(automatic > threshold).item()
            ),
            "edited_support_pixels": int(
                torch.count_nonzero(edited > threshold).item()
            ),
            "manual_add_pixels": int(
                torch.count_nonzero(manual_add > threshold).item()
            ),
            "manual_erase_pixels": int(
                torch.count_nonzero(manual_erase > threshold).item()
            ),
            "protected_core_restore_pixels": int(
                torch.count_nonzero(protected_core_restore > threshold).item()
            ),
            "final_support_pixels": int(
                torch.count_nonzero(final_mask > threshold).item()
            ),
            "mandatory_core_lock_passed": core_lock_passed,
            "outside_processing_support_is_zero": outside_support_passed,
            "user_may_add_transition": True,
            "user_may_erase_transition": True,
            "user_may_erase_mandatory_core": False,
            "preview_legend": {
                "green": "manual add",
                "magenta": "manual erase",
                "yellow": "attempted core erase restored by mandatory lock",
            },
            "gate_passed": bool(core_lock_passed and outside_support_passed),
        }
        return (
            final_mask,
            manual_add,
            manual_erase,
            preview.clamp(0.0, 1.0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalOriginalHighFrequencyTransfer:
    """Add only aligned source high-frequency detail to an AI image.

    The source low-frequency image is never mixed into the generated image.
    Reflective Gaussian padding prevents a padded crop boundary from becoming
    a false high-frequency line.  Strong structural edges are attenuated so a
    slightly shifted source contour cannot create a duplicated bright/dark
    outline on the generated subject.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "mask": ("MASK",),
                "high_frequency_radius": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.5, "max": 32.0, "step": 0.25},
                ),
                "detail_strength": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
                "edge_protection_start": (
                    "FLOAT",
                    {"default": 0.035, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "edge_protection_end": (
                    "FLOAT",
                    {"default": 0.120, "min": 0.001, "max": 1.0, "step": 0.005},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "detail_restored_image",
        "high_frequency_preview",
        "effective_detail_mask",
        "report_json",
    )
    FUNCTION = "transfer"
    CATEGORY = "face local edit/3 detail"

    def transfer(
        self,
        generated_image,
        original_image,
        mask,
        high_frequency_radius=2.0,
        detail_strength=0.35,
        edge_protection_start=0.035,
        edge_protection_end=0.120,
    ):
        generated = _image_batch(generated_image).to(
            device="cpu", dtype=torch.float32
        )
        original = _image_batch(original_image).to(
            device="cpu", dtype=torch.float32
        )
        if generated.shape != original.shape:
            raise ValueError(
                "generated_image and original_image must have identical dimensions"
            )
        height, width = int(generated.shape[1]), int(generated.shape[2])
        selected = _mask_batch(mask, height, width).to(
            device="cpu", dtype=torch.float32
        )
        radius = float(high_frequency_radius)
        strength = float(detail_strength)
        edge_start = float(edge_protection_start)
        edge_end = float(edge_protection_end)
        if radius < 0.5:
            raise ValueError("high_frequency_radius must be at least 0.5")
        if not 0.0 <= strength <= 1.5:
            raise ValueError("detail_strength must be between 0 and 1.5")
        if not 0.0 <= edge_start < edge_end <= 1.0:
            raise ValueError(
                "edge protection must satisfy 0 <= start < end <= 1"
            )

        original_np = original.numpy()
        low_frequency_np = ndimage.gaussian_filter(
            original_np,
            sigma=(0.0, radius, radius, 0.0),
            mode="reflect",
        )
        high_frequency = torch.from_numpy(
            original_np - low_frequency_np
        ).to(dtype=torch.float32)
        high_magnitude = torch.amax(
            torch.abs(high_frequency), dim=-1
        )
        edge_position = torch.clamp(
            (high_magnitude - edge_start) / (edge_end - edge_start),
            0.0,
            1.0,
        )
        smooth_edge_position = (
            edge_position
            * edge_position
            * (3.0 - 2.0 * edge_position)
        )
        structure_protection = 1.0 - smooth_edge_position
        effective_mask = selected * structure_protection
        restored = torch.clamp(
            generated
            + high_frequency
            * effective_mask.unsqueeze(-1)
            * strength,
            0.0,
            1.0,
        )
        preview = torch.clamp(
            0.5 + high_frequency * strength,
            0.0,
            1.0,
        )

        selected_pixels = int(
            torch.count_nonzero(selected > 0.001).item()
        )
        effective_pixels = int(
            torch.count_nonzero(effective_mask > 0.001).item()
        )
        outside_selected_exact = bool(
            torch.equal(
                restored[selected <= 0.001],
                generated[selected <= 0.001],
            )
        )
        report = {
            "algorithm": (
                "source-high-frequency-only-reflect-padding-"
                "structure-edge-protected-v1"
            ),
            "formula": (
                "generated + strength * "
                "(original - gaussian_blur_reflect(original)) * "
                "independent_mask * structure_edge_protection"
            ),
            "low_frequency_transfer": False,
            "high_frequency_radius": radius,
            "detail_strength": strength,
            "edge_protection_start": edge_start,
            "edge_protection_end": edge_end,
            "selected_pixels": selected_pixels,
            "effective_detail_pixels": effective_pixels,
            "outside_selected_mask_is_exact_generated": (
                outside_selected_exact
            ),
            "crop_boundary_padding_mode": "reflect",
            "visual_review_still_required": True,
            "gate_passed": outside_selected_exact,
        }
        return (
            restored,
            preview,
            effective_mask,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalSyntheticSkinMicrotexture:
    """Synthesize source-like skin microtexture without source coordinates.

    The compatibility route retains only radial spectral statistics and uses
    fresh target-space phase.  The JW-OLA route samples source skin residual
    patches and rebuilds them with jittered Hann overlap-add.  Neither route
    transfers source coordinates, low-frequency colour, or lighting, so source
    eyes, nose, lips, and face contour cannot be copied into the edited face.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "source_skin_patch": ("IMAGE",),
                "mask": ("MASK",),
                "source_iod": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 4096.0, "step": 1.0},
                ),
                "target_iod": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 4096.0, "step": 1.0},
                ),
                "detail_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
                "quantile_blend": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 0.80, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 20260725,
                        "min": 0,
                        "max": 0x7FFFFFFFFFFFFFFF,
                    },
                ),
            },
            "optional": {
                "synthesis_mode": (
                    ["radial_random_phase_v1", "jwola_microtexture_v1"],
                    {"default": "radial_random_phase_v1"},
                ),
                "dog_sigma_small": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.25, "max": 8.0, "step": 0.05},
                ),
                "dog_sigma_large": (
                    "FLOAT",
                    {"default": 2.20, "min": 0.50, "max": 32.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "detail_restored_image",
        "synthesized_texture_preview",
        "source_band_preview",
        "effective_detail_mask",
        "report_json",
    )
    FUNCTION = "synthesize"
    CATEGORY = "face local edit/3 detail"

    @staticmethod
    def _srgb_to_linear(rgb):
        return np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )

    @staticmethod
    def _linear_to_srgb(rgb):
        return np.where(
            rgb <= 0.0031308,
            rgb * 12.92,
            1.055 * np.maximum(rgb, 0.0) ** (1.0 / 2.4) - 0.055,
        )

    @classmethod
    def _rgb_to_lab(cls, rgb):
        linear = cls._srgb_to_linear(np.clip(rgb, 0.0, 1.0))
        matrix = np.array(
            (
                (0.4124564, 0.3575761, 0.1804375),
                (0.2126729, 0.7151522, 0.0721750),
                (0.0193339, 0.1191920, 0.9503041),
            ),
            dtype=np.float32,
        )
        xyz = linear @ matrix.T
        white = np.array((0.95047, 1.0, 1.08883), dtype=np.float32)
        ratio = xyz / white
        delta = 6.0 / 29.0
        f = np.where(
            ratio > delta**3,
            np.cbrt(ratio),
            ratio / (3.0 * delta**2) + 4.0 / 29.0,
        )
        lab = np.empty_like(f)
        lab[..., 0] = 116.0 * f[..., 1] - 16.0
        lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
        lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
        return lab

    @classmethod
    def _lab_to_rgb(cls, lab):
        fy = (lab[..., 0] + 16.0) / 116.0
        fx = fy + lab[..., 1] / 500.0
        fz = fy - lab[..., 2] / 200.0
        delta = 6.0 / 29.0
        f = np.stack((fx, fy, fz), axis=-1)
        ratio = np.where(
            f > delta,
            f**3,
            3.0 * delta**2 * (f - 4.0 / 29.0),
        )
        white = np.array((0.95047, 1.0, 1.08883), dtype=np.float32)
        xyz = ratio * white
        matrix = np.array(
            (
                (3.2404542, -1.5371385, -0.4985314),
                (-0.9692660, 1.8760108, 0.0415560),
                (0.0556434, -0.2040259, 1.0572252),
            ),
            dtype=np.float32,
        )
        linear = xyz @ matrix.T
        return np.clip(cls._linear_to_srgb(linear), 0.0, 1.0)

    @staticmethod
    def _extract_band(source_l, sigma_small, sigma_large):
        small = ndimage.gaussian_filter(
            source_l,
            sigma=sigma_small,
            mode="reflect",
        )
        large = ndimage.gaussian_filter(
            source_l,
            sigma=sigma_large,
            mode="reflect",
        )
        band = small - large
        return band - float(np.mean(band))

    @staticmethod
    def _radial_profile(texture):
        height, width = texture.shape
        window = np.outer(
            np.hanning(height),
            np.hanning(width),
        ).astype(np.float32)
        power = np.abs(np.fft.rfft2(texture * window)) ** 2
        fy = np.fft.fftfreq(height)[:, None]
        fx = np.fft.rfftfreq(width)[None, :]
        radius = np.sqrt(fy * fy + fx * fx)
        bin_count = max(32, min(height, width) // 2)
        edges = np.linspace(0.0, np.sqrt(0.5), bin_count + 1)
        sums, _ = np.histogram(radius, bins=edges, weights=power)
        counts, _ = np.histogram(radius, bins=edges)
        radial_power = sums / np.maximum(counts, 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        valid = counts > 0
        return centers[valid], radial_power[valid]

    @staticmethod
    def _synthesize_radial(shape, frequencies, radial_power, rms, seed):
        height, width = shape
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(shape).astype(np.float32)
        noise_spectrum = np.fft.rfft2(noise)
        phase = noise_spectrum / np.maximum(
            np.abs(noise_spectrum),
            1e-12,
        )
        fy = np.fft.fftfreq(height)[:, None]
        fx = np.fft.rfftfreq(width)[None, :]
        radius = np.sqrt(fy * fy + fx * fx)
        amplitude = np.sqrt(
            np.maximum(
                np.interp(
                    radius,
                    frequencies,
                    radial_power,
                    left=0.0,
                    right=0.0,
                ),
                0.0,
            )
        )
        synthesized = np.fft.irfft2(
            amplitude * phase,
            s=shape,
        ).real
        synthesized -= float(np.mean(synthesized))
        synthesized_rms = float(
            np.sqrt(np.mean(synthesized * synthesized))
        )
        if synthesized_rms <= 1e-12:
            raise ValueError("source skin patch has no usable spectral energy")
        return (
            synthesized * (rms / synthesized_rms)
        ).astype(np.float32)

    @staticmethod
    def _quantile_blend(synthesized, source_texture, rms, blend):
        if blend <= 0.0:
            return synthesized
        quantiles = np.linspace(0.005, 0.995, 1025)
        source_values = np.quantile(source_texture, quantiles)
        synthesized_values = np.quantile(synthesized, quantiles)
        synthesized_values = np.maximum.accumulate(synthesized_values)
        unique_values, unique_indices = np.unique(
            synthesized_values,
            return_index=True,
        )
        mapped = np.interp(
            synthesized,
            unique_values,
            source_values[unique_indices],
            left=source_values[0],
            right=source_values[-1],
        )
        result = synthesized * (1.0 - blend) + mapped * blend
        result -= float(np.mean(result))
        result_rms = float(np.sqrt(np.mean(result * result)))
        if result_rms <= 1e-12:
            raise ValueError("quantile matching produced zero-energy texture")
        return (result * (rms / result_rms)).astype(np.float32)

    @staticmethod
    def _local_rms_samples(texture):
        short_side = min(texture.shape)
        window = max(5, int(round(0.12 * short_side)))
        if window % 2 == 0:
            window += 1
        maximum = short_side - 2
        if maximum % 2 == 0:
            maximum -= 1
        window = max(3, min(window, maximum))
        local_mean = ndimage.uniform_filter(
            texture,
            size=window,
            mode="reflect",
        )
        local_square_mean = ndimage.uniform_filter(
            texture * texture,
            size=window,
            mode="reflect",
        )
        local_rms = np.sqrt(
            np.maximum(
                local_square_mean - local_mean * local_mean,
                0.0,
            )
        )
        border = window // 2
        interior = local_rms[border:-border, border:-border]
        samples = interior[
            np.isfinite(interior) & (interior > 1e-9)
        ]
        if samples.size < 64:
            raise ValueError("too few valid source local-RMS samples")
        low, high = np.quantile(samples, [0.05, 0.95])
        samples = np.clip(samples, low, high)
        mean = float(np.mean(samples))
        if mean <= 1e-12:
            raise ValueError("source local-RMS distribution has zero energy")
        return (samples / mean).astype(np.float32), int(window)

    @staticmethod
    def _source_patch_candidates(
        source_texture,
        patch_size,
        seed,
        maximum_candidates=8192,
    ):
        height, width = source_texture.shape
        if min(height, width) < patch_size * 2:
            raise ValueError(
                "source skin patch must be at least two micro-patches wide"
            )
        rng = np.random.default_rng(seed)
        possible_y = height - patch_size + 1
        possible_x = width - patch_size + 1
        total = possible_y * possible_x
        if total <= maximum_candidates:
            flat = np.arange(total, dtype=np.int64)
        else:
            flat = rng.choice(
                total,
                size=maximum_candidates,
                replace=False,
            )
        ys = flat // possible_x
        xs = flat % possible_x
        rms = np.empty(flat.shape[0], dtype=np.float32)
        for index, (y, x) in enumerate(zip(ys, xs)):
            patch = source_texture[
                y : y + patch_size,
                x : x + patch_size,
            ]
            centered = patch - float(np.mean(patch))
            rms[index] = float(np.sqrt(np.mean(centered * centered)))
        valid = rms > 1e-9
        if np.count_nonzero(valid) < 32:
            raise ValueError("too few non-empty source micro-patches")
        low, high = np.quantile(rms[valid], [0.05, 0.95])
        accepted = valid & (rms >= low) & (rms <= high)
        if np.count_nonzero(accepted) < 32:
            raise ValueError("too few accepted source micro-patches")
        return (
            np.stack((ys[accepted], xs[accepted]), axis=1),
            {
                "sampled_patch_count": int(flat.size),
                "accepted_patch_count": int(np.count_nonzero(accepted)),
                "accepted_rms_low": float(low),
                "accepted_rms_high": float(high),
            },
        )

    @staticmethod
    def _hann_window(patch_size):
        one_dimensional = np.hanning(patch_size).astype(np.float32)
        return np.maximum(
            np.outer(one_dimensional, one_dimensional),
            1e-6,
        ).astype(np.float32)

    @classmethod
    def _synthesize_jwola(cls, source_texture, shape, seed):
        patch_size = 24
        overlap_ratio = 0.60
        jitter_ratio = 0.35
        hop = max(1, int(round(patch_size * (1.0 - overlap_ratio))))
        jitter = int(round(hop * jitter_ratio))
        candidates, candidate_report = cls._source_patch_candidates(
            source_texture,
            patch_size,
            seed + 1000,
        )
        height, width = shape
        pad = patch_size * 2
        canvas_height = height + pad * 2
        canvas_width = width + pad * 2
        accumulated = np.zeros(
            (canvas_height, canvas_width),
            dtype=np.float32,
        )
        weights = np.zeros_like(accumulated)
        window = cls._hann_window(patch_size)
        rng = np.random.default_rng(seed)
        usage = np.zeros(candidates.shape[0], dtype=np.int32)
        placed = 0
        for anchor_y in range(
            0,
            canvas_height - patch_size + 1,
            hop,
        ):
            for anchor_x in range(
                0,
                canvas_width - patch_size + 1,
                hop,
            ):
                y = int(
                    np.clip(
                        anchor_y + rng.integers(-jitter, jitter + 1),
                        0,
                        canvas_height - patch_size,
                    )
                )
                x = int(
                    np.clip(
                        anchor_x + rng.integers(-jitter, jitter + 1),
                        0,
                        canvas_width - patch_size,
                    )
                )
                candidate_id = int(rng.integers(0, candidates.shape[0]))
                source_y, source_x = candidates[candidate_id]
                patch = source_texture[
                    source_y : source_y + patch_size,
                    source_x : source_x + patch_size,
                ].copy()
                patch -= float(np.mean(patch))
                if bool(rng.integers(0, 2)):
                    patch = np.fliplr(patch)
                accumulated[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += patch * window
                weights[
                    y : y + patch_size,
                    x : x + patch_size,
                ] += window
                usage[candidate_id] += 1
                placed += 1
        crop_weights = weights[pad : pad + height, pad : pad + width]
        if float(np.min(crop_weights)) <= 0.0:
            raise ValueError(
                "jittered overlap-add left uncovered target pixels"
            )
        texture = (
            accumulated[pad : pad + height, pad : pad + width]
            / crop_weights
        )
        texture -= float(np.mean(texture))

        normalized_rms, local_rms_window = cls._local_rms_samples(
            source_texture
        )
        envelope_rng = np.random.default_rng(seed + 2000)
        envelope = envelope_rng.standard_normal(shape).astype(np.float32)
        correlation_sigma = max(12.0, 0.08 * min(shape))
        envelope = ndimage.gaussian_filter(
            envelope,
            sigma=correlation_sigma,
            mode="reflect",
        )
        envelope_std = float(np.std(envelope))
        if envelope_std <= 1e-12:
            raise ValueError("coarse RMS envelope has zero variance")
        envelope = (envelope - float(np.mean(envelope))) / envelope_std
        envelope = np.quantile(
            normalized_rms,
            ndtr(envelope),
        ).astype(np.float32)
        envelope /= max(float(np.mean(envelope)), 1e-12)
        envelope = np.clip(envelope, 0.4, 1.8)
        envelope /= max(float(np.mean(envelope)), 1e-12)
        texture = (texture * envelope).astype(np.float32)
        report = {
            **candidate_report,
            "patch_size": patch_size,
            "overlap_ratio": overlap_ratio,
            "hop": hop,
            "jitter_ratio": jitter_ratio,
            "jitter_pixels": jitter,
            "placed_patch_count": placed,
            "unique_source_patches_used": int(np.count_nonzero(usage)),
            "maximum_single_patch_usage": int(np.max(usage)),
            "local_rms_window": local_rms_window,
            "envelope_correlation_sigma": correlation_sigma,
            "envelope_std": float(np.std(envelope)),
        }
        return texture, report

    @staticmethod
    def _preview(texture):
        robust = float(np.quantile(np.abs(texture), 0.995))
        scale = 0.45 / max(robust, 1e-8)
        gray = np.clip(0.5 + texture * scale, 0.0, 1.0)
        return np.repeat(gray[..., None], 3, axis=-1)

    def synthesize(
        self,
        generated_image,
        source_skin_patch,
        mask,
        source_iod=1.0,
        target_iod=1.0,
        detail_strength=0.75,
        quantile_blend=0.65,
        seed=20260725,
        synthesis_mode="radial_random_phase_v1",
        dog_sigma_small=0.65,
        dog_sigma_large=2.20,
    ):
        generated = _image_batch(generated_image).to(
            device="cpu",
            dtype=torch.float32,
        )
        source = _image_batch(source_skin_patch).to(
            device="cpu",
            dtype=torch.float32,
        )
        height, width = int(generated.shape[1]), int(generated.shape[2])
        selected = _mask_batch(mask, height, width).to(
            device="cpu",
            dtype=torch.float32,
        )
        source_iod_value = float(source_iod)
        target_iod_value = float(target_iod)
        strength = float(detail_strength)
        blend = float(quantile_blend)
        mode = str(synthesis_mode)
        sigma_small = float(dog_sigma_small)
        sigma_large = float(dog_sigma_large)
        if source_iod_value <= 0.0 or target_iod_value <= 0.0:
            raise ValueError("source_iod and target_iod must be positive")
        scale = target_iod_value / source_iod_value
        if not 0.10 <= scale <= 10.0:
            raise ValueError("target_iod/source_iod must be between 0.10 and 10.0")
        if not 0.0 <= strength <= 1.5:
            raise ValueError("detail_strength must be between 0.0 and 1.5")
        if not 0.0 <= blend <= 0.80:
            raise ValueError("quantile_blend must be between 0.0 and 0.80")
        if not 0.0 < sigma_small < sigma_large:
            raise ValueError(
                "DoG sigmas must satisfy 0 < small < large"
            )
        if mode not in {
            "radial_random_phase_v1",
            "jwola_microtexture_v1",
        }:
            raise ValueError(f"unsupported synthesis mode: {mode}")
        if source.shape[0] not in (1, generated.shape[0]):
            raise ValueError(
                "source_skin_patch batch must be 1 or match generated_image"
            )

        restored_batches = []
        texture_previews = []
        source_previews = []
        effective_masks = []
        batch_reports = []
        for batch_index in range(generated.shape[0]):
            generated_np = generated[batch_index].numpy()
            source_index = 0 if source.shape[0] == 1 else batch_index
            source_np = source[source_index].numpy()
            source_l = self._rgb_to_lab(source_np)[..., 0] / 100.0
            if abs(scale - 1.0) >= 1e-6:
                source_l = ndimage.zoom(
                    source_l,
                    zoom=scale,
                    order=3,
                    mode="reflect",
                )
            if min(source_l.shape) < 64:
                raise ValueError(
                    "resampled source skin patch short side must be at least 64 px"
                )
            if max(source_l.shape) > 4096:
                raise ValueError(
                    "resampled source skin patch long side must not exceed 4096 px"
                )
            source_texture = self._extract_band(
                source_l,
                sigma_small,
                sigma_large,
            )
            source_rms = float(
                np.sqrt(np.mean(source_texture * source_texture))
            )
            if source_rms <= 1e-6:
                raise ValueError(
                    "source skin patch is too smooth to provide usable texture"
                )
            frequencies, radial_power = self._radial_profile(
                source_texture
            )
            selected_np = selected[batch_index].numpy()
            support_y, support_x = np.nonzero(selected_np > 0.001)
            restored_np = generated_np.copy()
            texture_preview_np = np.full_like(generated_np, 0.5)
            effective_np = np.zeros((height, width), dtype=np.float32)
            clip_ratio = 0.0
            synthesis_report = None
            calibrated_rms = None
            if support_x.size:
                x1 = max(int(support_x.min()) - 16, 0)
                y1 = max(int(support_y.min()) - 16, 0)
                x2 = min(int(support_x.max()) + 17, width)
                y2 = min(int(support_y.max()) + 17, height)
                patch_shape = (y2 - y1, x2 - x1)
                if mode == "jwola_microtexture_v1":
                    texture, synthesis_report = self._synthesize_jwola(
                        source_texture,
                        patch_shape,
                        int(seed) + batch_index,
                    )
                else:
                    texture = self._synthesize_radial(
                        patch_shape,
                        frequencies,
                        radial_power,
                        source_rms,
                        int(seed) + batch_index,
                    )
                    texture = self._quantile_blend(
                        texture,
                        source_texture,
                        source_rms,
                        blend,
                    )
                patch_mask = selected_np[y1:y2, x1:x2]
                target_patch = generated_np[y1:y2, x1:x2]
                target_lab = self._rgb_to_lab(target_patch)
                target_l = target_lab[..., 0] / 100.0
                headroom = np.minimum(target_l, 1.0 - target_l)
                headroom_weight = np.clip(headroom / 0.08, 0.0, 1.0)
                weighted_texture = (
                    texture * patch_mask * headroom_weight
                )
                if mode == "jwola_microtexture_v1":
                    evaluation = (
                        (patch_mask >= 0.95)
                        & (headroom_weight >= 0.95)
                    )
                    if np.count_nonzero(evaluation) < 256:
                        evaluation = (
                            (patch_mask >= 0.50)
                            & (headroom_weight >= 0.50)
                        )
                    if np.count_nonzero(evaluation) < 64:
                        raise ValueError(
                            "strict skin interior is too small for RMS calibration"
                        )
                    current_rms = float(
                        np.sqrt(
                            np.mean(
                                weighted_texture[evaluation]
                                * weighted_texture[evaluation]
                            )
                        )
                    )
                    if current_rms <= 1e-12:
                        raise ValueError("weighted texture has zero energy")
                    target_rms = source_rms * strength
                    calibration_alpha = target_rms / current_rms
                    raw_delta = calibration_alpha * weighted_texture
                    calibrated_rms = float(
                        np.sqrt(np.mean(raw_delta[evaluation] ** 2))
                    )
                else:
                    calibration_alpha = strength
                    raw_delta = weighted_texture * strength
                safe_limit = np.maximum(
                    headroom - (1.0 / 1024.0),
                    0.0,
                )
                delta = np.clip(raw_delta, -safe_limit, safe_limit)
                clip_ratio = float(
                    np.count_nonzero(
                        np.abs(delta - raw_delta) > 1e-12
                    )
                    / max(np.count_nonzero(patch_mask > 0.001), 1)
                )
                target_lab[..., 0] = (target_l + delta) * 100.0
                converted = self._lab_to_rgb(target_lab)
                exact_outside = patch_mask <= 0.001
                converted[exact_outside] = target_patch[exact_outside]
                restored_np[y1:y2, x1:x2] = converted
                texture_preview_np[y1:y2, x1:x2] = self._preview(
                    texture * patch_mask * calibration_alpha
                )
                effective_np[y1:y2, x1:x2] = (
                    patch_mask * headroom_weight
                )
                bbox = [x1, y1, x2, y2]
            else:
                bbox = None

            source_previews.append(
                torch.from_numpy(self._preview(source_texture))
            )
            restored_tensor = torch.from_numpy(
                restored_np.astype(np.float32)
            )
            outside_exact = bool(
                torch.equal(
                    restored_tensor[selected[batch_index] <= 0.001],
                    generated[batch_index][
                        selected[batch_index] <= 0.001
                    ],
                )
            )
            restored_batches.append(restored_tensor)
            texture_previews.append(
                torch.from_numpy(texture_preview_np.astype(np.float32))
            )
            effective_masks.append(torch.from_numpy(effective_np))
            batch_reports.append(
                {
                    "batch_index": int(batch_index),
                    "source_texture_rms": source_rms,
                    "synthesis_bbox_xyxy": bbox,
                    "selected_pixels": int(
                        np.count_nonzero(selected_np > 0.001)
                    ),
                    "clip_ratio": clip_ratio,
                    "synthesis_report": synthesis_report,
                    "calibration_alpha": float(calibration_alpha),
                    "calibrated_preclip_rms": calibrated_rms,
                    "outside_mask_is_exact_generated": outside_exact,
                }
            )

        report = {
            "algorithm": mode,
            "source_coordinate_transfer": False,
            "source_phase_transfer": False,
            "source_low_frequency_transfer": False,
            "source_iod": source_iod_value,
            "target_iod": target_iod_value,
            "iod_scale": scale,
            "dog_sigma_small": sigma_small,
            "dog_sigma_large": sigma_large,
            "detail_strength": strength,
            "quantile_blend": blend,
            "quantile_blend_applied": (
                mode == "radial_random_phase_v1"
            ),
            "seed": int(seed),
            "batch": batch_reports,
            "manual_mask_is_authoritative": True,
            "visual_review_still_required": True,
            "gate_passed": all(
                item["outside_mask_is_exact_generated"]
                for item in batch_reports
            ),
        }
        return (
            torch.stack(restored_batches, dim=0),
            torch.stack(texture_previews, dim=0),
            torch.stack(source_previews, dim=0),
            torch.stack(effective_masks, dim=0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalOuterBoundaryFeather:
    """Feather the crop perimeter and the exterior of the selected seam.

    Threshold selection and manual editing decide *where* AI pixels are
    definitely retained.  A narrow outward-only ramp is added immediately
    outside that selection, so every selected pixel stays at its existing alpha
    while the surrounding source image receives a short, smooth transition.
    The existing four-sided crop-perimeter ramp remains independent.  Each
    requested crop feather width is capped to the available distance before the
    mandatory head/face core, so restoring that core to alpha 1 cannot introduce
    an internal hard edge.

    New nodes default to the operator-facing directional profile.  For a
    portrait crop, top/bottom use 10% and left/right use 5%; for a landscape
    crop, left/right use 10% and top/bottom use 5%; a square uses 10% on all
    four sides.  Orientation is inferred from the crop dimensions.  The outer
    workflow exposes only four direct physical-side percentages.  Older
    profile, adjustment, and multiplier inputs remain inside the node solely
    for saved-workflow compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "corrected_mask": ("MASK",),
                "mandatory_core_mask": ("MASK",),
                "processing_support_mask": ("MASK",),
                "selection": ("FACE_LOCAL_SELECTION",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
            },
            "optional": {
                "axis_feather_profile": (
                    [
                        "long_edge_10_short_edge_5",
                        "long_axis_10_short_axis_5",
                        "manual_per_side",
                        "legacy_uniform_10",
                    ],
                    {"default": "long_axis_10_short_axis_5"},
                ),
                "feather_width_mode": (
                    ["crop_short_side_ratio", "fixed_pixels"],
                    {"default": "crop_short_side_ratio"},
                ),
                "outer_feather_crop_ratio": (
                    "FLOAT",
                    {"default": 0.100, "min": 0.001, "max": 0.30, "step": 0.001},
                ),
                "fixed_feather_pixels": (
                    "INT",
                    {"default": 24, "min": 1, "max": 512, "step": 1},
                ),
                "minimum_feather_pixels": (
                    "INT",
                    {"default": 24, "min": 1, "max": 512, "step": 1},
                ),
                "maximum_feather_pixels": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 8192, "step": 1},
                ),
                "top_feather_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "bottom_feather_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "left_feather_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "right_feather_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "feather_at_original_image_boundary": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "outer_feather_face_ratio": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.30,
                        "step": 0.005,
                    },
                ),
                "internal_difference_feather_crop_ratio": (
                    "FLOAT",
                    {
                        "default": 0.006,
                        "min": 0.0,
                        "max": 0.03,
                        "step": 0.001,
                    },
                ),
                "long_edge_feather_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "short_edge_feather_percent": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "top_feather_adjustment_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5},
                ),
                "bottom_feather_adjustment_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5},
                ),
                "left_feather_adjustment_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5},
                ),
                "right_feather_adjustment_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5},
                ),
                "top_feather_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "bottom_feather_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "left_feather_percent": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "right_feather_percent": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = (
        "final_feathered_mask",
        "outer_boundary_feather_ramp",
        "report_json",
    )
    FUNCTION = "feather"
    CATEGORY = "face local edit/2 mask"

    def feather(
        self,
        corrected_mask,
        mandatory_core_mask,
        processing_support_mask,
        selection,
        x,
        y,
        width,
        height,
        outer_feather_face_ratio=0.0,
        axis_feather_profile="long_axis_10_short_axis_5",
        feather_width_mode="crop_short_side_ratio",
        outer_feather_crop_ratio=0.100,
        fixed_feather_pixels=24,
        minimum_feather_pixels=24,
        maximum_feather_pixels=4096,
        top_feather_multiplier=1.0,
        bottom_feather_multiplier=1.0,
        left_feather_multiplier=1.0,
        right_feather_multiplier=1.0,
        feather_at_original_image_boundary=False,
        internal_difference_feather_crop_ratio=0.006,
        long_edge_feather_percent=10.0,
        short_edge_feather_percent=5.0,
        top_feather_adjustment_percent=0.0,
        bottom_feather_adjustment_percent=0.0,
        left_feather_adjustment_percent=0.0,
        right_feather_adjustment_percent=0.0,
        top_feather_percent=10.0,
        bottom_feather_percent=10.0,
        left_feather_percent=5.0,
        right_feather_percent=5.0,
    ):
        local_width, local_height = int(width), int(height)
        if local_width <= 0 or local_height <= 0:
            raise ValueError("width and height must be positive")
        corrected_tensor = _mask_batch(
            corrected_mask, local_height, local_width
        )
        core_tensor = _mask_batch(
            mandatory_core_mask, local_height, local_width
        )
        support_tensor = _mask_batch(
            processing_support_mask, local_height, local_width
        )
        corrected = corrected_tensor[0].numpy().astype(np.float32)
        core = core_tensor[0].numpy() > (0.5 / 255.0)
        support = support_tensor[0].numpy() > (0.5 / 255.0)
        if not np.any(support):
            raise ValueError("processing_support_mask is empty")

        face = _selection_face(selection)
        gx1, gy1, gx2, gy2 = (float(v) for v in face["bbox_xyxy"])
        face_short = min(gx2 - gx1, gy2 - gy1)
        if face_short <= 0:
            raise ValueError("Selected face has invalid dimensions")
        mode = str(feather_width_mode)
        if mode not in ("crop_short_side_ratio", "fixed_pixels"):
            raise ValueError(
                "feather_width_mode must be crop_short_side_ratio or fixed_pixels"
            )
        feather_ratio = float(outer_feather_crop_ratio)
        if not 0.001 <= feather_ratio <= 0.30:
            raise ValueError(
                "outer_feather_crop_ratio must be between 0.001 and 0.30"
            )
        minimum_pixels = int(minimum_feather_pixels)
        maximum_pixels = int(maximum_feather_pixels)
        if minimum_pixels < 1 or maximum_pixels < minimum_pixels:
            raise ValueError(
                "feather pixel limits must satisfy 1 <= minimum <= maximum"
            )
        legacy_face_ratio = float(outer_feather_face_ratio)
        if legacy_face_ratio > 0.0:
            if legacy_face_ratio > 0.30:
                raise ValueError(
                    "outer_feather_face_ratio must not exceed 0.30"
                )
            feather_pixels = max(
                1, int(round(face_short * legacy_face_ratio))
            )
            width_source = "legacy_face_short_side_ratio"
        elif mode == "fixed_pixels":
            feather_pixels = int(fixed_feather_pixels)
            if feather_pixels < 1:
                raise ValueError("fixed_feather_pixels must be positive")
            width_source = "fixed_pixels"
        else:
            crop_short = min(local_width, local_height)
            raw_pixels = int(round(crop_short * feather_ratio))
            feather_pixels = min(
                maximum_pixels, max(minimum_pixels, raw_pixels)
            )
            width_source = "crop_short_side_ratio_clamped"

        original_height = int(selection["image_height"])
        original_width = int(selection["image_width"])
        (
            original_x0,
            original_y0,
            original_x1,
            original_y1,
            local_x0,
            local_y0,
            local_x1,
            local_y1,
        ) = _crop_intersection(
            int(x),
            int(y),
            local_width,
            local_height,
            original_width,
            original_height,
        )
        valid_original = np.zeros((local_height, local_width), dtype=bool)
        valid_original[local_y0:local_y1, local_x0:local_x1] = True

        user_multipliers = {
            "top": float(top_feather_multiplier),
            "bottom": float(bottom_feather_multiplier),
            "left": float(left_feather_multiplier),
            "right": float(right_feather_multiplier),
        }
        if any(
            value < 0.0 or value > 3.0
            for value in user_multipliers.values()
        ):
            raise ValueError("per-side feather multipliers must be within 0.0–3.0")
        profile = str(axis_feather_profile)
        if profile not in (
            "long_edge_10_short_edge_5",
            "long_axis_10_short_axis_5",
            "manual_per_side",
            "legacy_uniform_10",
        ):
            raise ValueError(
                "axis_feather_profile must be long_edge_10_short_edge_5, "
                "long_axis_10_short_axis_5, manual_per_side, or "
                "legacy_uniform_10"
            )
        long_edge_percent = float(long_edge_feather_percent)
        short_edge_percent = float(short_edge_feather_percent)
        if not 0.0 <= long_edge_percent <= 30.0:
            raise ValueError("long_edge_feather_percent must be within 0.0–30.0")
        if not 0.0 <= short_edge_percent <= 30.0:
            raise ValueError("short_edge_feather_percent must be within 0.0–30.0")
        side_adjustments = {
            "top": float(top_feather_adjustment_percent),
            "bottom": float(bottom_feather_adjustment_percent),
            "left": float(left_feather_adjustment_percent),
            "right": float(right_feather_adjustment_percent),
        }
        if any(
            value < -30.0 or value > 30.0
            for value in side_adjustments.values()
        ):
            raise ValueError(
                "per-side feather percentage-point adjustments must be within "
                "-30.0–30.0"
            )
        direct_side_percent = {
            "top": float(top_feather_percent),
            "bottom": float(bottom_feather_percent),
            "left": float(left_feather_percent),
            "right": float(right_feather_percent),
        }
        if any(
            not 0.0 <= value <= 30.0
            for value in direct_side_percent.values()
        ):
            raise ValueError(
                "per-side feather percentages must be within 0.0–30.0"
            )

        edge_classes = {
            "top": "uniform",
            "bottom": "uniform",
            "left": "uniform",
            "right": "uniform",
        }
        side_base_percent = {
            "top": feather_ratio * 100.0,
            "bottom": feather_ratio * 100.0,
            "left": feather_ratio * 100.0,
            "right": feather_ratio * 100.0,
        }
        if profile == "long_edge_10_short_edge_5":
            width_source = "physical_edge_percent_clamped"
            if local_width > local_height:
                edge_classes = {
                    "top": "long_edge",
                    "bottom": "long_edge",
                    "left": "short_edge",
                    "right": "short_edge",
                }
                axis_orientation = "landscape"
            elif local_height > local_width:
                edge_classes = {
                    "top": "short_edge",
                    "bottom": "short_edge",
                    "left": "long_edge",
                    "right": "long_edge",
                }
                axis_orientation = "portrait"
            else:
                edge_classes = {
                    "top": "equal_edge",
                    "bottom": "equal_edge",
                    "left": "equal_edge",
                    "right": "equal_edge",
                }
                axis_orientation = "square"
            side_base_percent = {
                side: (
                    short_edge_percent
                    if edge_class == "short_edge"
                    else long_edge_percent
                )
                for side, edge_class in edge_classes.items()
            }
            axis_multipliers = {
                side: 1.0 for side in ("top", "bottom", "left", "right")
            }
        elif profile == "long_axis_10_short_axis_5":
            width_source = "long_axis_per_side_percent"
            if local_width > local_height:
                edge_classes = {
                    "top": "short_axis",
                    "bottom": "short_axis",
                    "left": "long_axis",
                    "right": "long_axis",
                }
                axis_orientation = "landscape"
            elif local_height > local_width:
                edge_classes = {
                    "top": "long_axis",
                    "bottom": "long_axis",
                    "left": "short_axis",
                    "right": "short_axis",
                }
                axis_orientation = "portrait"
            else:
                edge_classes = {
                    "top": "equal_axis",
                    "bottom": "equal_axis",
                    "left": "equal_axis",
                    "right": "equal_axis",
                }
                axis_orientation = "square"
            side_base_percent = {
                side: (
                    short_edge_percent
                    if edge_class == "short_axis"
                    else long_edge_percent
                )
                for side, edge_class in edge_classes.items()
            }
            axis_multipliers = {
                side: 1.0 for side in ("top", "bottom", "left", "right")
            }
        else:
            axis_multipliers = {
                "top": 1.0,
                "bottom": 1.0,
                "left": 1.0,
                "right": 1.0,
            }
            axis_orientation = "manual"
        multipliers = {
            side: user_multipliers[side] * axis_multipliers[side]
            for side in ("top", "bottom", "left", "right")
        }
        portrait_defaults = {
            "top": 10.0,
            "bottom": 10.0,
            "left": 5.0,
            "right": 5.0,
        }
        uses_automatic_side_defaults = (
            profile == "long_axis_10_short_axis_5"
            and direct_side_percent == portrait_defaults
        )
        side_effective_percent = {
            side: (
                direct_side_percent[side]
                if profile == "long_axis_10_short_axis_5"
                and not uses_automatic_side_defaults
                else side_base_percent[side] + side_adjustments[side]
            )
            for side in ("top", "bottom", "left", "right")
        }
        if profile in (
            "long_edge_10_short_edge_5",
            "long_axis_10_short_axis_5",
        ) and any(
            value < 0.0 or value > 30.0
            for value in side_effective_percent.values()
        ):
            raise ValueError(
                "effective per-side feather percentages must be within 0.0–30.0"
            )
        touches_original_boundary = {
            "top": original_y0 == 0,
            "bottom": original_y1 == original_height,
            "left": original_x0 == 0,
            "right": original_x1 == original_width,
        }
        allow_image_boundary = bool(feather_at_original_image_boundary)
        side_pixels = {}
        side_pixel_basis = {}
        for side, multiplier in multipliers.items():
            suppressed = (
                touches_original_boundary[side] and not allow_image_boundary
            )
            direct_percent = side_effective_percent[side]
            if suppressed or multiplier == 0.0 or (
                profile == "long_edge_10_short_edge_5" and direct_percent == 0.0
            ):
                side_pixel_basis[side] = 0
                side_pixels[side] = 0
                continue

            if profile in (
                "long_edge_10_short_edge_5",
                "long_axis_10_short_axis_5",
            ):
                axis_length = (
                    local_height if side in ("top", "bottom") else local_width
                )
                side_pixel_basis[side] = axis_length
                if mode == "crop_short_side_ratio" and legacy_face_ratio <= 0.0:
                    raw_side_pixels = int(round(axis_length * direct_percent / 100.0))
                else:
                    reference_percent = max(long_edge_percent, 0.001)
                    raw_side_pixels = int(
                        round(feather_pixels * direct_percent / reference_percent)
                    )
                if raw_side_pixels <= 0:
                    side_pixels[side] = 0
                elif (
                    profile == "long_axis_10_short_axis_5"
                    and mode == "fixed_pixels"
                ):
                    # Preserve the original fixed-pixel profile: a 10 px
                    # request becomes 10 px on the long direction and 5 px on
                    # the short direction, without the ratio-mode minimum.
                    side_pixels[side] = min(
                        maximum_pixels, max(1, raw_side_pixels)
                    )
                else:
                    side_pixels[side] = min(
                        maximum_pixels,
                        max(minimum_pixels, raw_side_pixels),
                    )
            else:
                side_pixel_basis[side] = feather_pixels
                side_pixels[side] = max(
                    1, int(round(feather_pixels * multiplier))
                )

        requested_side_pixels = dict(side_pixels)
        core_clearance_pixels = {
            "top": None,
            "bottom": None,
            "left": None,
            "right": None,
        }
        core_clearance_cap_applied = {
            "top": False,
            "bottom": False,
            "left": False,
            "right": False,
        }
        insufficient_crop_context_sides = []
        active_core = core & support & valid_original
        if np.any(active_core):
            core_y, core_x = np.nonzero(active_core)
            core_clearance_pixels = {
                # Distances use the same one-based convention as the
                # smoothstep ramps below.  At exactly ``pixels`` the ramp is
                # already 1.0, so the first mandatory-core pixel remains
                # continuous with its outer neighbour.
                "top": int(core_y.min() - local_y0 + 1),
                "bottom": int(local_y1 - core_y.max()),
                "left": int(core_x.min() - local_x0 + 1),
                "right": int(local_x1 - core_x.max()),
            }
            for side in ("top", "bottom", "left", "right"):
                requested = int(side_pixels[side])
                if requested <= 0:
                    continue
                clearance = max(1, int(core_clearance_pixels[side]))
                if clearance < requested:
                    side_pixels[side] = clearance
                    core_clearance_cap_applied[side] = True
                if clearance <= 1:
                    insufficient_crop_context_sides.append(side)

        yy, xx = np.indices((local_height, local_width), dtype=np.float32)
        side_distances = {
            "top": yy - float(local_y0) + 1.0,
            "bottom": float(local_y1) - yy,
            "left": xx - float(local_x0) + 1.0,
            "right": float(local_x1) - xx,
        }
        feather_ramp = np.ones(
            (local_height, local_width), dtype=np.float32
        )
        boundary_zone = np.zeros(
            (local_height, local_width), dtype=bool
        )
        for side, pixels in side_pixels.items():
            if pixels <= 0:
                continue
            distance = side_distances[side]
            side_ramp = _smoothstep_numpy(0.0, float(pixels), distance)
            feather_ramp = np.minimum(feather_ramp, side_ramp)
            boundary_zone |= valid_original & (distance <= float(pixels))
        feather_ramp[~valid_original] = 0.0

        internal_ratio = float(internal_difference_feather_crop_ratio)
        if not 0.0 <= internal_ratio <= 0.03:
            raise ValueError(
                "internal_difference_feather_crop_ratio must be between 0.0 and 0.03"
            )
        internal_pixels = (
            max(2, min(96, int(round(min(local_width, local_height) * internal_ratio))))
            if internal_ratio > 0.0
            else 0
        )
        corrected_support = corrected > (0.5 / 255.0)
        internal_transition = np.zeros_like(corrected, dtype=np.float32)
        if internal_pixels > 0 and np.any(corrected_support):
            # Only the true exterior is softened.  A user-erased enclosed hole
            # remains erased instead of being silently painted back by the
            # feather operation.
            filled_support = ndimage.binary_fill_holes(corrected_support)
            exterior = ~filled_support
            outside_distance = ndimage.distance_transform_edt(
                ~corrected_support
            ).astype(np.float32)
            transition_zone = (
                exterior
                & support
                & valid_original
                & (outside_distance > 0.0)
                & (outside_distance <= float(internal_pixels))
            )
            internal_transition[transition_zone] = 1.0 - _smoothstep_numpy(
                0.0,
                float(internal_pixels),
                outside_distance[transition_zone],
            )

        seam_feathered = np.maximum(corrected, internal_transition)
        feathered = seam_feathered * feather_ramp
        final = np.maximum(feathered, core.astype(np.float32))
        final[~support] = 0.0
        final[~valid_original] = 0.0

        core_lock_passed = bool(np.all(final[core] == 1.0))
        outside_support_passed = bool(np.all(final[~support] == 0.0))
        outside_original_passed = bool(
            np.all(final[~valid_original] == 0.0)
        )
        report = {
            "algorithm": (
                "post-manual-correction-outer-crop-boundary-"
                "core-clearance-capped-smoothstep-feather-v2"
            ),
            "formula": (
                "clip_to_support(max(mandatory_core, "
                "corrected_mask * outer_boundary_feather_ramp))"
            ),
            "crop_xywh": [
                int(x),
                int(y),
                local_width,
                local_height,
            ],
            "original_intersection_xyxy": [
                original_x0,
                original_y0,
                original_x1,
                original_y1,
            ],
            "local_intersection_xyxy": [
                local_x0,
                local_y0,
                local_x1,
                local_y1,
            ],
            "feather_width_mode": mode,
            "feather_width_source": width_source,
            "outer_feather_crop_ratio": feather_ratio,
            "fixed_feather_pixels": int(fixed_feather_pixels),
            "minimum_feather_pixels": minimum_pixels,
            "maximum_feather_pixels": maximum_pixels,
            "axis_feather_profile": profile,
            "axis_orientation": axis_orientation,
            "physical_edge_class": edge_classes,
            "long_edge_feather_percent": long_edge_percent,
            "short_edge_feather_percent": short_edge_percent,
            "per_side_base_percent": side_base_percent,
            "per_side_adjustment_percentage_points": side_adjustments,
            "per_side_direct_percent": direct_side_percent,
            "per_side_direct_percent_mode": (
                "automatic_long_axis_10_short_axis_5"
                if uses_automatic_side_defaults
                else "manual_physical_sides"
            ),
            "per_side_effective_percent": side_effective_percent,
            "axis_profile_multiplier": axis_multipliers,
            "per_side_user_multiplier": user_multipliers,
            "per_side_multiplier": multipliers,
            "per_side_pixel_basis": side_pixel_basis,
            "touches_original_image_boundary": touches_original_boundary,
            "feather_at_original_image_boundary": allow_image_boundary,
            "requested_feather_pixels": requested_side_pixels,
            "effective_feather_pixels": side_pixels,
            "mandatory_core_clearance_pixels": core_clearance_pixels,
            "core_clearance_cap_applied": core_clearance_cap_applied,
            "insufficient_crop_context_sides": insufficient_crop_context_sides,
            "legacy_outer_feather_face_ratio": legacy_face_ratio,
            "outer_feather_pixels": feather_pixels,
            "internal_difference_feather_crop_ratio": internal_ratio,
            "internal_difference_feather_pixels": internal_pixels,
            "pixels": {
                "corrected_support": int(
                    np.count_nonzero(corrected > (0.5 / 255.0))
                ),
                "boundary_zone": int(np.count_nonzero(boundary_zone)),
                "partially_feathered": int(
                    np.count_nonzero(
                        (final > (0.5 / 255.0)) & (final < 1.0)
                    )
                ),
                "internal_transition": int(
                    np.count_nonzero(internal_transition > (0.5 / 255.0))
                ),
                "mandatory_core": int(np.count_nonzero(core)),
                "final_support": int(
                    np.count_nonzero(final > (0.5 / 255.0))
                ),
            },
            "contracts": {
                "feather_is_after_manual_correction": True,
                "crop_perimeter_and_internal_seam_are_independent": True,
                "internal_feather_is_outward_only": True,
                "internal_selected_pixels_are_unchanged_before_crop_perimeter_ramp": bool(
                    np.all(seam_feathered[corrected_support] == corrected[corrected_support])
                ),
                "enclosed_manual_erase_is_not_refilled": True,
                "feather_ramp_reaches_full_ai_before_mandatory_core": bool(
                    not insufficient_crop_context_sides
                ),
                "mandatory_core_relocked_to_full_ai": core_lock_passed,
                "outside_processing_support_is_zero": outside_support_passed,
                "outside_original_image_is_zero": outside_original_passed,
            },
            "gate_passed": bool(
                core_lock_passed
                and outside_support_passed
                and outside_original_passed
            ),
            "visual_review_still_required": True,
        }
        return (
            torch.from_numpy(final.copy()).unsqueeze(0),
            torch.from_numpy(feather_ramp.copy()).unsqueeze(0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalAdaptiveDifferenceMask:
    """Create the internal writeback alpha from two local images.

    The complete semantic head/face core is always exact AI output.  Outside
    that core, a smoothed perceptual difference field adds only coherent
    high-confidence changes and continuously returns weak differences to the
    source.  This node consumes workflow-internal images; it is not an external
    third-image input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_local": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "selection": ("FACE_LOCAL_SELECTION",),
                "processing_support_mask": ("MASK",),
                "full_target_mask": ("MASK",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "low_threshold": (
                    "FLOAT",
                    {"default": 0.035, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "high_threshold": (
                    "FLOAT",
                    {"default": 0.120, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "difference_blur_face_ratio": (
                    "FLOAT",
                    {"default": 0.015, "min": 0.0, "max": 0.10, "step": 0.005},
                ),
                "connectivity_close_face_ratio": (
                    "FLOAT",
                    {"default": 0.040, "min": 0.0, "max": 0.20, "step": 0.005},
                ),
            },
            "optional": {
                "generated_face_mask": ("MASK",),
                "core_guard_face_ratio": (
                    "FLOAT",
                    {
                        "default": ADAPTIVE_CORE_GUARD_FACE_RATIO,
                        "min": 0.0,
                        "max": 0.05,
                        "step": 0.001,
                    },
                ),
                "compatibility_band_face_ratio": (
                    "FLOAT",
                    {
                        "default": ADAPTIVE_COMPATIBILITY_BAND_FACE_RATIO,
                        "min": 0.0,
                        "max": 0.20,
                        "step": 0.005,
                    },
                ),
                "difference_full_strength_face_ratio": (
                    "FLOAT",
                    {
                        "default": ADAPTIVE_DIFFERENCE_FULL_STRENGTH_FACE_RATIO,
                        "min": 0.0,
                        "max": 0.60,
                        "step": 0.01,
                    },
                ),
                "difference_outer_limit_face_ratio": (
                    "FLOAT",
                    {
                        "default": ADAPTIVE_DIFFERENCE_OUTER_LIMIT_FACE_RATIO,
                        "min": 0.0,
                        "max": 0.80,
                        "step": 0.01,
                    },
                ),
                "boundary_suppression_face_ratio": (
                    "FLOAT",
                    {
                        "default": ADAPTIVE_BOUNDARY_SUPPRESSION_FACE_RATIO,
                        "min": 0.0,
                        "max": 0.20,
                        "step": 0.005,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "automatic_union_mask",
        "mandatory_head_face_core",
        "adaptive_transition_mask",
        "red_cyan_difference_preview",
        "processing_support_passthrough",
        "report_json",
    )
    FUNCTION = "build"
    CATEGORY = "face local edit/2 mask"

    def build(
        self,
        source_local,
        generated_local,
        selection,
        processing_support_mask,
        full_target_mask,
        x,
        y,
        width,
        height,
        low_threshold=0.035,
        high_threshold=0.120,
        difference_blur_face_ratio=0.015,
        connectivity_close_face_ratio=0.040,
        core_guard_face_ratio=ADAPTIVE_CORE_GUARD_FACE_RATIO,
        compatibility_band_face_ratio=ADAPTIVE_COMPATIBILITY_BAND_FACE_RATIO,
        difference_full_strength_face_ratio=(
            ADAPTIVE_DIFFERENCE_FULL_STRENGTH_FACE_RATIO
        ),
        difference_outer_limit_face_ratio=(
            ADAPTIVE_DIFFERENCE_OUTER_LIMIT_FACE_RATIO
        ),
        boundary_suppression_face_ratio=(
            ADAPTIVE_BOUNDARY_SUPPRESSION_FACE_RATIO
        ),
        generated_face_mask=None,
    ):
        source = _image_batch(source_local).to(device="cpu", dtype=torch.float32)
        generated = _image_batch(generated_local).to(device="cpu", dtype=torch.float32)
        if source.shape != generated.shape:
            raise ValueError("source_local and generated_local must have identical dimensions")
        local_height, local_width = int(source.shape[1]), int(source.shape[2])
        px, py, planned_width, planned_height = (
            int(x),
            int(y),
            int(width),
            int(height),
        )
        if (local_width, local_height) != (planned_width, planned_height):
            raise ValueError(
                "Local image dimensions do not match the semantic crop coordinates"
            )
        low, high = float(low_threshold), float(high_threshold)
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")

        support_tensor = _mask_batch(
            processing_support_mask, local_height, local_width
        )
        support = support_tensor[0].numpy() > (0.5 / 255.0)
        if not np.any(support):
            raise ValueError("processing_support_mask is empty")

        face = _selection_face(selection)
        gx1, gy1, gx2, gy2 = (float(v) for v in face["bbox_xyxy"])
        local_bbox = (gx1 - px, gy1 - py, gx2 - px, gy2 - py)
        face_short = min(gx2 - gx1, gy2 - gy1)
        if face_short <= 0:
            raise ValueError("Selected face has invalid dimensions")

        original_height = int(selection["image_height"])
        original_width = int(selection["image_width"])
        full_target = _mask_batch(
            full_target_mask, original_height, original_width
        )[0].numpy() > 0.001
        (
            original_x0,
            original_y0,
            original_x1,
            original_y1,
            local_x0,
            local_y0,
            local_x1,
            local_y1,
        ) = _crop_intersection(
            px,
            py,
            local_width,
            local_height,
            original_width,
            original_height,
        )
        valid_original = np.zeros((local_height, local_width), dtype=bool)
        valid_original[local_y0:local_y1, local_x0:local_x1] = True
        local_target = np.zeros((local_height, local_width), dtype=bool)
        local_target[local_y0:local_y1, local_x0:local_x1] = full_target[
            original_y0:original_y1,
            original_x0:original_x1,
        ]
        filled_semantic = ndimage.binary_fill_holes(local_target)
        compact_face = _compact_complete_face_core(
            (local_height, local_width), local_bbox
        )
        generated_face_core = np.zeros(
            (local_height, local_width), dtype=bool
        )
        if generated_face_mask is not None:
            generated_face_core = (
                _mask_batch(
                    generated_face_mask,
                    local_height,
                    local_width,
                )[0].numpy()
                > (0.5 / 255.0)
            )
            if not np.any(generated_face_core):
                raise ValueError("generated_face_mask is empty")
            generated_face_core &= valid_original
        mandatory_core_unclipped = (
            filled_semantic | compact_face | generated_face_core
        ) & valid_original
        core_guard = max(1, int(round(face_short * float(core_guard_face_ratio))))
        mandatory_core_unclipped = ndimage.binary_dilation(
            mandatory_core_unclipped,
            structure=_disk_structure(core_guard),
        )
        mandatory_core_unclipped &= valid_original
        core_outside_support_pixels = int(
            np.count_nonzero(mandatory_core_unclipped & ~support)
        )
        if core_outside_support_pixels:
            raise ValueError(
                "Complete semantic head/face core extends outside processing support; "
                "increase the semantic crop context instead of clipping the head"
            )
        mandatory_core = mandatory_core_unclipped & support

        compatibility_radius = max(
            1,
            int(round(face_short * float(compatibility_band_face_ratio))),
        )
        outside_core_distance = ndimage.distance_transform_edt(
            ~mandatory_core
        ).astype(np.float32)
        compatibility = 1.0 - _smoothstep_numpy(
            0.0,
            float(compatibility_radius),
            outside_core_distance,
        )
        compatibility[mandatory_core] = 1.0
        compatibility[~support] = 0.0
        compatibility[~valid_original] = 0.0

        source_lab = _srgb_to_lab(source)
        generated_lab = _srgb_to_lab(generated)
        delta = torch.abs(source_lab - generated_lab)
        luminance = delta[:, 0] / 100.0
        chroma = torch.linalg.vector_norm(delta[:, 1:3], dim=1) / (
            255.0 * math.sqrt(2.0)
        )
        raw_difference = (
            luminance * 0.70 + chroma * 0.30
        ).clamp(0.0, 1.0)[0].numpy().astype(np.float32)
        blur_sigma = max(
            1.0, face_short * float(difference_blur_face_ratio)
        )
        smoothed = ndimage.gaussian_filter(
            raw_difference, sigma=blur_sigma, mode="nearest"
        ).astype(np.float32)

        close_radius = max(
            1, int(round(face_short * float(connectivity_close_face_ratio)))
        )
        difference_full_strength_radius = max(
            compatibility_radius + 1,
            int(
                round(
                    face_short * float(difference_full_strength_face_ratio)
                )
            ),
        )
        difference_outer_limit_radius = max(
            difference_full_strength_radius + 1,
            int(
                round(face_short * float(difference_outer_limit_face_ratio))
            ),
        )
        transition_support = (
            support
            & valid_original
            & (
                outside_core_distance
                <= float(difference_outer_limit_radius)
            )
        )
        low_region = (smoothed >= low) & transition_support
        closed = ndimage.binary_closing(
            low_region,
            structure=_disk_structure(close_radius),
        )
        high_seeds = (smoothed >= high) & transition_support
        labels, count = ndimage.label(closed & transition_support)
        if count:
            seed_labels = np.unique(labels[high_seeds])
            seed_labels = seed_labels[seed_labels != 0]
            coherent = (
                np.isin(labels, seed_labels)
                if seed_labels.size
                else np.zeros_like(support)
            )
            coherent = (
                ndimage.binary_fill_holes(coherent) & transition_support
            )
        else:
            coherent = np.zeros_like(support)

        difference_strength = _smoothstep_numpy(low, high, smoothed)
        transition = np.where(coherent, difference_strength, 0.0).astype(
            np.float32
        )
        outer_fade = 1.0 - _smoothstep_numpy(
            float(difference_full_strength_radius),
            float(difference_outer_limit_radius),
            outside_core_distance,
        )
        transition *= outer_fade
        boundary_radius = max(
            2,
            int(
                round(face_short * float(boundary_suppression_face_ratio))
            ),
        )
        valid_distance = ndimage.distance_transform_edt(valid_original).astype(
            np.float32
        )
        boundary_ramp = _smoothstep_numpy(
            0.0,
            float(boundary_radius),
            valid_distance,
        )
        transition *= boundary_ramp
        transition[mandatory_core] = 0.0
        transition[~support] = 0.0
        transition[~valid_original] = 0.0
        automatic = np.maximum.reduce(
            (
                mandatory_core.astype(np.float32),
                compatibility.astype(np.float32),
                transition,
            )
        ).astype(np.float32)
        automatic[mandatory_core] = 1.0
        automatic[~support] = 0.0
        automatic[~valid_original] = 0.0

        luminance_weights = torch.tensor(
            (0.299, 0.587, 0.114), dtype=torch.float32
        ).view(1, 1, 1, 3)
        source_gray = torch.sum(source * luminance_weights, dim=-1, keepdim=True)
        generated_gray = torch.sum(
            generated * luminance_weights, dim=-1, keepdim=True
        )
        red_cyan = torch.cat(
            (source_gray, generated_gray, generated_gray), dim=-1
        ).clamp(0.0, 1.0)

        valid_support = support & valid_original
        boundary_distance = ndimage.distance_transform_edt(valid_support)
        boundary_probe = valid_support & (boundary_distance <= 2.0)
        boundary_alpha = automatic[boundary_probe]
        boundary_high_fraction = (
            float(np.mean(boundary_alpha >= 0.20))
            if boundary_alpha.size
            else 0.0
        )
        target_semantic_pixels = int(np.count_nonzero(local_target))
        target_semantic_covered_pixels = int(
            np.count_nonzero(local_target & mandatory_core)
        )
        target_semantic_coverage_fraction = (
            float(target_semantic_covered_pixels / target_semantic_pixels)
            if target_semantic_pixels
            else 0.0
        )
        report = {
            "algorithm": (
                "compact-complete-head-face-core-compatibility-band-"
                "bounded-adaptive-difference-v4"
            ),
            "external_image_inputs": 0,
            "source_and_generated_are_internal_workflow_results": True,
            "crop_xywh": [px, py, local_width, local_height],
            "original_intersection_xyxy": [
                original_x0,
                original_y0,
                original_x1,
                original_y1,
            ],
            "local_intersection_xyxy": [
                local_x0,
                local_y0,
                local_x1,
                local_y1,
            ],
            "padded_context_pixels": int(
                local_width * local_height
                - (original_x1 - original_x0) * (original_y1 - original_y0)
            ),
            "local_face_bbox_xyxy": [float(v) for v in local_bbox],
            "thresholds": {"low": low, "high": high},
            "derived_pixels": {
                "difference_blur_sigma": blur_sigma,
                "connectivity_close_radius": close_radius,
                "core_guard_radius": core_guard,
                "compatibility_band_radius": compatibility_radius,
                "difference_full_strength_radius": (
                    difference_full_strength_radius
                ),
                "difference_outer_limit_radius": (
                    difference_outer_limit_radius
                ),
                "boundary_suppression_radius": boundary_radius,
            },
            "pixels": {
                "processing_support": int(np.count_nonzero(support)),
                "valid_original": int(np.count_nonzero(valid_original)),
                "target_semantic_head_hair": target_semantic_pixels,
                "target_semantic_head_hair_covered": target_semantic_covered_pixels,
                "generated_face_core": int(
                    np.count_nonzero(generated_face_core)
                ),
                "mandatory_head_face_core": int(np.count_nonzero(mandatory_core)),
                "compatibility_band": int(
                    np.count_nonzero(
                        compatibility > (0.5 / 255.0)
                    )
                ),
                "adaptive_transition": int(
                    np.count_nonzero(transition > (0.5 / 255.0))
                ),
                "automatic_union": int(
                    np.count_nonzero(automatic > (0.5 / 255.0))
                ),
            },
            "contracts": {
                "complete_head_face_core_is_full_ai": bool(
                    np.all(automatic[mandatory_core] == 1.0)
                ),
                "semantic_head_hair_coverage_fraction": target_semantic_coverage_fraction,
                "semantic_head_hair_is_fully_covered": bool(
                    target_semantic_pixels > 0
                    and target_semantic_covered_pixels == target_semantic_pixels
                ),
                "core_is_semantic_head_hair_plus_broad_ears_chin_upper_neck": True,
                "generated_face_contour_is_in_mandatory_core": bool(
                    not np.any(generated_face_core & ~mandatory_core)
                ),
                "mandatory_core_is_compact_not_rectangular": True,
                "compatibility_band_is_appended_by_union": True,
                "adaptive_difference_is_appended_by_union": True,
                "outside_processing_support_is_zero": bool(
                    np.all(automatic[~support] == 0.0)
                ),
                "outside_original_image_is_zero": bool(
                    np.all(automatic[~valid_original] == 0.0)
                ),
                "adaptive_difference_is_zero_outside_outer_limit": bool(
                    np.all(
                        transition[
                            outside_core_distance
                            > float(difference_outer_limit_radius)
                        ]
                        == 0.0
                    )
                ),
            },
            "boundary_high_alpha_fraction": boundary_high_fraction,
            "boundary_warning": (
                "enlarge crop or manually erase transition at the processing boundary"
                if boundary_high_fraction > 0.01
                else None
            ),
            "gate_passed": bool(
                target_semantic_pixels > 0
                and target_semantic_covered_pixels == target_semantic_pixels
                and np.all(automatic[mandatory_core] == 1.0)
                and np.all(automatic[~support] == 0.0)
                and np.all(automatic[~valid_original] == 0.0)
            ),
            "visual_review_still_required": True,
        }
        return (
            torch.from_numpy(automatic.copy()).unsqueeze(0),
            torch.from_numpy(mandatory_core.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(transition.copy()).unsqueeze(0),
            red_cyan,
            support_tensor,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalThresholdDifferenceMask:
    """Build a literal integer RGB-difference threshold mask.

    The threshold is the maximum absolute 8-bit RGB channel difference for
    each aligned pixel.  Every pixel at or above the selected threshold is
    retained from the AI result.  ``pure_difference`` uses that literal
    partition alone.  ``difference_plus_mandatory_core`` preserves the legacy
    complete semantic head/face union for existing workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_local": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "processing_support_mask": ("MASK",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "threshold_level": (
                    "INT",
                    {"default": 7, "min": 6, "max": 8, "step": 1},
                ),
            },
            "optional": {
                "selection": ("FACE_LOCAL_SELECTION",),
                "full_target_mask": ("MASK",),
                "generated_face_mask": ("MASK",),
                "core_guard_face_ratio": (
                    "FLOAT",
                    {
                        "default": ADAPTIVE_CORE_GUARD_FACE_RATIO,
                        "min": 0.0,
                        "max": 0.05,
                        "step": 0.001,
                    },
                ),
                "mask_mode": (
                    [
                        "difference_plus_mandatory_core",
                        "pure_difference",
                    ],
                    {"default": "difference_plus_mandatory_core"},
                ),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "automatic_union_mask",
        "mandatory_head_face_core",
        "threshold_difference_transition",
        "red_cyan_difference_preview",
        "processing_support_passthrough",
        "report_json",
    )
    FUNCTION = "build"
    CATEGORY = "face local edit/2 mask"

    def build(
        self,
        source_local,
        generated_local,
        selection=None,
        processing_support_mask=None,
        full_target_mask=None,
        x=0,
        y=0,
        width=0,
        height=0,
        threshold_level=7,
        core_guard_face_ratio=ADAPTIVE_CORE_GUARD_FACE_RATIO,
        mask_mode="difference_plus_mandatory_core",
        generated_face_mask=None,
    ):
        source = _image_batch(source_local).to(device="cpu", dtype=torch.float32)
        generated = _image_batch(generated_local).to(
            device="cpu", dtype=torch.float32
        )
        if source.shape != generated.shape:
            raise ValueError(
                "source_local and generated_local must have identical dimensions"
            )

        local_height, local_width = int(source.shape[1]), int(source.shape[2])
        px, py, planned_width, planned_height = (
            int(x),
            int(y),
            int(width),
            int(height),
        )
        if (local_width, local_height) != (planned_width, planned_height):
            raise ValueError(
                "Local image dimensions do not match the semantic crop coordinates"
            )

        threshold = int(threshold_level)
        if threshold not in (6, 7, 8):
            raise ValueError("threshold_level must be exactly 6, 7, or 8")
        mode = str(mask_mode)
        if mode not in (
            "difference_plus_mandatory_core",
            "pure_difference",
        ):
            raise ValueError(
                "mask_mode must be difference_plus_mandatory_core "
                "or pure_difference"
            )
        if processing_support_mask is None:
            raise ValueError("processing_support_mask is required")

        support_tensor = _mask_batch(
            processing_support_mask, local_height, local_width
        )
        support = support_tensor[0].numpy() > (0.5 / 255.0)
        if not np.any(support):
            raise ValueError("processing_support_mask is empty")

        if mode == "difference_plus_mandatory_core":
            if selection is None or full_target_mask is None:
                raise ValueError(
                    "selection and full_target_mask are required in "
                    "difference_plus_mandatory_core mode"
                )
            face = _selection_face(selection)
            gx1, gy1, gx2, gy2 = (float(v) for v in face["bbox_xyxy"])
            local_bbox = (gx1 - px, gy1 - py, gx2 - px, gy2 - py)
            face_short = min(gx2 - gx1, gy2 - gy1)
            if face_short <= 0:
                raise ValueError("Selected face has invalid dimensions")
            original_height = int(selection["image_height"])
            original_width = int(selection["image_width"])
        else:
            local_bbox = None
            face_short = 0.0
            original_height = max(0, py + local_height)
            original_width = max(0, px + local_width)

        (
            original_x0,
            original_y0,
            original_x1,
            original_y1,
            local_x0,
            local_y0,
            local_x1,
            local_y1,
        ) = _crop_intersection(
            px,
            py,
            local_width,
            local_height,
            original_width,
            original_height,
        )
        valid_original = np.zeros((local_height, local_width), dtype=bool)
        valid_original[local_y0:local_y1, local_x0:local_x1] = True
        local_target = np.zeros((local_height, local_width), dtype=bool)
        generated_face_core = np.zeros(
            (local_height, local_width), dtype=bool
        )
        core_guard = 0
        if mode == "difference_plus_mandatory_core":
            full_target = _mask_batch(
                full_target_mask, original_height, original_width
            )[0].numpy() > 0.001
            local_target[local_y0:local_y1, local_x0:local_x1] = full_target[
                original_y0:original_y1,
                original_x0:original_x1,
            ]
            filled_semantic = ndimage.binary_fill_holes(local_target)
            compact_face = _compact_complete_face_core(
                (local_height, local_width), local_bbox
            )
            if generated_face_mask is not None:
                generated_face_core = (
                    _mask_batch(
                        generated_face_mask, local_height, local_width
                    )[0].numpy()
                    > (0.5 / 255.0)
                )
                if not np.any(generated_face_core):
                    raise ValueError("generated_face_mask is empty")
                generated_face_core &= valid_original
            mandatory_core_unclipped = (
                filled_semantic | compact_face | generated_face_core
            ) & valid_original
            core_guard = max(
                1, int(round(face_short * float(core_guard_face_ratio)))
            )
            mandatory_core_unclipped = ndimage.binary_dilation(
                mandatory_core_unclipped,
                structure=_disk_structure(core_guard),
            )
            mandatory_core_unclipped &= valid_original
            core_outside_support_pixels = int(
                np.count_nonzero(mandatory_core_unclipped & ~support)
            )
            if core_outside_support_pixels:
                raise ValueError(
                    "Complete semantic head/face core extends outside processing "
                    "support; increase the semantic crop context instead of "
                    "clipping the head"
                )
            mandatory_core = mandatory_core_unclipped & support
        else:
            mandatory_core = np.zeros(
                (local_height, local_width), dtype=bool
            )

        source_u8 = torch.round(source.clamp(0.0, 1.0) * 255.0).to(
            torch.int16
        )
        generated_u8 = torch.round(
            generated.clamp(0.0, 1.0) * 255.0
        ).to(torch.int16)
        channel_delta = torch.abs(generated_u8 - source_u8)
        max_channel_delta = torch.amax(channel_delta, dim=-1)[0].numpy()
        threshold_difference = (
            (max_channel_delta >= threshold) & support & valid_original
        )
        transition = threshold_difference & ~mandatory_core
        if mode == "pure_difference":
            automatic = threshold_difference.astype(np.float32)
        else:
            automatic = (
                threshold_difference | mandatory_core
            ).astype(np.float32)

        luminance_weights = torch.tensor(
            (0.299, 0.587, 0.114), dtype=torch.float32
        ).view(1, 1, 1, 3)
        source_gray = torch.sum(
            source * luminance_weights, dim=-1, keepdim=True
        )
        generated_gray = torch.sum(
            generated * luminance_weights, dim=-1, keepdim=True
        )
        red_cyan = torch.cat(
            (source_gray, generated_gray, generated_gray), dim=-1
        ).clamp(0.0, 1.0)

        target_semantic_pixels = int(np.count_nonzero(local_target))
        target_semantic_covered_pixels = int(
            np.count_nonzero(local_target & mandatory_core)
        )
        legacy_core_contract_applies = (
            mode == "difference_plus_mandatory_core"
        )
        algorithm = (
            "literal-max-rgb-difference-pure-v2"
            if mode == "pure_difference"
            else (
                "mandatory-complete-head-face-core-union-"
                "literal-max-rgb-difference-v2"
            )
        )
        report = {
            "algorithm": algorithm,
            "mask_mode": mode,
            "external_image_inputs": 0,
            "source_and_generated_are_internal_workflow_results": True,
            "difference_metric": "max(abs(ai_u8_rgb - source_u8_rgb))",
            "threshold_level": threshold,
            "threshold_contract": (
                "all aligned pixels with max-channel difference >= threshold "
                "retain AI output"
            ),
            "crop_xywh": [px, py, local_width, local_height],
            "original_intersection_xyxy": [
                original_x0,
                original_y0,
                original_x1,
                original_y1,
            ],
            "local_intersection_xyxy": [
                local_x0,
                local_y0,
                local_x1,
                local_y1,
            ],
            "local_face_bbox_xyxy": (
                [float(v) for v in local_bbox]
                if local_bbox is not None
                else None
            ),
            "derived_pixels": {"core_guard_radius": core_guard},
            "pixels": {
                "processing_support": int(np.count_nonzero(support)),
                "valid_original": int(np.count_nonzero(valid_original)),
                "target_semantic_head_hair": target_semantic_pixels,
                "target_semantic_head_hair_covered": (
                    target_semantic_covered_pixels
                ),
                "mandatory_head_face_core": int(
                    np.count_nonzero(mandatory_core)
                ),
                "generated_face_core": int(
                    np.count_nonzero(generated_face_core)
                ),
                "threshold_difference_all": int(
                    np.count_nonzero(threshold_difference)
                ),
                "threshold_difference_transition": int(
                    np.count_nonzero(transition)
                ),
                "automatic_union": int(
                    np.count_nonzero(automatic > 0.5)
                ),
            },
            "contracts": {
                "complete_head_face_core_is_full_ai": bool(
                    not legacy_core_contract_applies
                    or np.all(automatic[mandatory_core] == 1.0)
                ),
                "semantic_head_hair_is_fully_covered": bool(
                    not legacy_core_contract_applies
                    or (
                        target_semantic_pixels > 0
                        and target_semantic_covered_pixels
                        == target_semantic_pixels
                    )
                ),
                "semantic_core_controls_final_range": (
                    legacy_core_contract_applies
                ),
                "generated_face_contour_is_in_mandatory_core": bool(
                    not legacy_core_contract_applies
                    or generated_face_mask is None
                    or np.all(mandatory_core[generated_face_core])
                ),
                "pure_difference_only": mode == "pure_difference",
                "threshold_is_in_supported_range": 6 <= threshold <= 8,
                "every_at_or_above_threshold_pixel_is_full_ai": bool(
                    np.all(automatic[threshold_difference] == 1.0)
                ),
                "below_threshold_outside_core_is_source": bool(
                    np.all(
                        automatic[
                            support
                            & valid_original
                            & ~mandatory_core
                            & (max_channel_delta < threshold)
                        ]
                        == 0.0
                    )
                ),
                "outside_processing_support_is_zero": bool(
                    np.all(automatic[~support] == 0.0)
                ),
                "outside_original_image_is_zero": bool(
                    np.all(automatic[~valid_original] == 0.0)
                ),
                "manual_editor_downstream_may_add_or_erase_transition": (
                    legacy_core_contract_applies
                ),
                "manual_editor_downstream_may_erase_mandatory_core": False,
            },
            "gate_passed": bool(
                (
                    not legacy_core_contract_applies
                    or (
                        target_semantic_pixels > 0
                        and target_semantic_covered_pixels
                        == target_semantic_pixels
                    )
                )
                and (
                    not legacy_core_contract_applies
                    or generated_face_mask is None
                    or np.all(mandatory_core[generated_face_core])
                )
                and np.all(automatic[mandatory_core] == 1.0)
                and np.all(automatic[threshold_difference] == 1.0)
                and np.all(automatic[~support] == 0.0)
                and np.all(automatic[~valid_original] == 0.0)
            ),
            "visual_review_still_required": True,
        }
        return (
            torch.from_numpy(automatic.copy()).unsqueeze(0),
            torch.from_numpy(mandatory_core.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(transition.astype(np.float32)).unsqueeze(0),
            red_cyan,
            support_tensor,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalPromptContract:
    PRESERVE = (
        "Make the edited face read as a clearly different individual, not as retouching, beauty work, "
        "makeup, aging, recoloring, or relighting. Preserve the exact head pose, gaze direction, "
        "expression intent, overall hairstyle, hair direction, hair length, hair color, ears, neck, body, "
        "clothing, composition, background, camera perspective, lighting direction, texture, sharpness, "
        "and image grain unless the user's instruction explicitly permits that item to change. The broad "
        "selected face/head region deliberately includes bangs or front hair, the hairline, ears, and upper "
        "neck: reconstruct everything inside it as one coherent photographic edit. Do not paste original "
        "hair back across the generated region, do not cut bangs at a mask edge, and do not create a straight "
        "or curved transition line across hair or skin. Introduce no unrelated objects or accessories."
    )
    REFERENCE_IDENTITY_GOAL = (
        "Make the edited face read as the same facial identity as the supplied identity reference. "
        "Do not merely create an arbitrary person who differs from the source, and do not average the "
        "source and reference identities. "
    )
    SOURCE_EYE_INTERIOR_PRESERVATION = (
        "Preserve Image 1's eye-interior appearance: iris color, diameter and texture, pupil size, shape "
        "and position, sclera color and fine texture, catchlight positions, local focus, and local noise. "
        "Do not import Image 2's irises, pupils, sclera pattern, gaze, or catchlights. Eye spacing, eye "
        "outline, and eyelid anatomy may follow the transferred facial geometry while respecting Image 1's "
        "gaze and expression constraints; do not replace the source eye interiors with smooth dark AI disks."
    )
    SOURCE_FACE_LIGHTING_PRESERVATION = (
        "Preserve Image 1's facial illumination: light direction and intensity, exposure, shadow shape "
        "and boundaries, highlight and catchlight placement, local contrast, color temperature, white "
        "balance, and camera response. Transfer identity geometry without importing Image 2's lighting "
        "or adding beauty lighting, symmetric oily highlights, or artificial under-eye shadows."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_instruction": (
                    "STRING",
                    {
                        "default": "Create a natural, coherent face with a clearly different overall identity while respecting all preservation constraints.",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "change_eyes": ("BOOLEAN", {"default": True}),
                "change_nose": ("BOOLEAN", {"default": True}),
                "change_mouth": ("BOOLEAN", {"default": True}),
                "change_jaw": ("BOOLEAN", {"default": False}),
                "change_brows": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prompt_mode": (
                    ["contract", "raw"],
                    {"default": "contract"},
                ),
                "identity_goal": (
                    ["different_from_source", "match_reference_identity"],
                    {"default": "different_from_source"},
                ),
                "preserve_source_eye_interior": ("BOOLEAN", {"default": False}),
                "preserve_source_face_lighting": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("contract_prompt", "structure_groups_json")
    FUNCTION = "build"
    CATEGORY = "face local edit/3 prompt"

    def build(
        self,
        edit_instruction,
        change_eyes,
        change_nose,
        change_mouth,
        change_jaw,
        change_brows,
        prompt_mode="contract",
        identity_goal="different_from_source",
        preserve_source_eye_interior=False,
        preserve_source_face_lighting=False,
    ):
        selected = [
            key
            for key, enabled in (
                ("eyes", change_eyes),
                ("nose", change_nose),
                ("mouth", change_mouth),
                ("jaw", change_jaw),
                ("brows", change_brows),
            )
            if bool(enabled)
        ]
        if len(selected) < 3:
            raise ValueError("At least three independent facial structure groups must be selected")
        raw_instruction = str(edit_instruction)
        if not raw_instruction.strip():
            raise ValueError("Edit instruction cannot be empty")
        selected_prompt_mode = str(prompt_mode)
        if selected_prompt_mode not in {"contract", "raw"}:
            raise ValueError(f"Unsupported prompt mode: {selected_prompt_mode}")
        instruction = raw_instruction.strip()
        structure_sentence = "Required visible structural changes: " + "; ".join(
            STRUCTURE_GROUPS[key] for key in selected
        ) + "."
        selected_identity_goal = str(identity_goal)
        if selected_identity_goal not in {"different_from_source", "match_reference_identity"}:
            raise ValueError(f"Unsupported identity goal: {selected_identity_goal}")
        if selected_prompt_mode == "raw":
            prompt = raw_instruction
        else:
            preserve_contract = self.PRESERVE
            if selected_identity_goal == "match_reference_identity":
                preserve_contract = preserve_contract.replace(
                    "Make the edited face read as a clearly different individual, not as retouching, beauty work, "
                    "makeup, aging, recoloring, or relighting. ",
                    self.REFERENCE_IDENTITY_GOAL,
                    1,
                )
            additions = []
            if bool(preserve_source_eye_interior):
                additions.append(self.SOURCE_EYE_INTERIOR_PRESERVATION)
            if bool(preserve_source_face_lighting):
                additions.append(self.SOURCE_FACE_LIGHTING_PRESERVATION)
            prompt_parts = [instruction.rstrip(), structure_sentence, preserve_contract, *additions]
            prompt = " ".join(part for part in prompt_parts if part)
        report = {
            "selected_groups": selected,
            "selected_group_count": len(selected),
            "minimum_required": 3,
            "prompt_mode": selected_prompt_mode,
            "automatic_contract_appended": selected_prompt_mode == "contract",
            "identity_goal": selected_identity_goal,
            "preserve_source_eye_interior": bool(preserve_source_eye_interior),
            "preserve_source_face_lighting": bool(preserve_source_face_lighting),
            "gate_passed": True,
        }
        return prompt, json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalImageSourceSelector:
    """Select the skin-sample reference before the single interactive crop.

    The direct reference is deliberately input 0. ComfyUI's normal BYPASS
    mapping therefore sends output 0 straight from input 0 when the complete
    stage-0 preprocessing group is bypassed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direct_identity_reference": ("IMAGE", {"lazy": True}),
                "gpt_identity_reference": ("IMAGE", {"lazy": True}),
                "enable_local_preprocess": ("BOOLEAN", {"default": False}),
                "enable_gpt_preprocess": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_reference",)
    FUNCTION = "select"
    CATEGORY = "face local edit/0 route"

    @classmethod
    def check_lazy_status(
        cls,
        direct_identity_reference=None,
        gpt_identity_reference=None,
        enable_local_preprocess=False,
        enable_gpt_preprocess=False,
    ):
        use_gpt_reference = bool(enable_local_preprocess) or bool(enable_gpt_preprocess)
        selected_name = (
            "gpt_identity_reference" if use_gpt_reference else "direct_identity_reference"
        )
        selected_value = (
            gpt_identity_reference if use_gpt_reference else direct_identity_reference
        )
        return [selected_name] if selected_value is None else []

    def select(
        self,
        direct_identity_reference=None,
        gpt_identity_reference=None,
        enable_local_preprocess=False,
        enable_gpt_preprocess=False,
    ):
        use_gpt_reference = bool(enable_local_preprocess) or bool(enable_gpt_preprocess)
        selected = gpt_identity_reference if use_gpt_reference else direct_identity_reference
        if selected is None:
            source = "GPT identity reference" if use_gpt_reference else "direct identity reference"
            raise ValueError(f"Selected {source} image is not connected")
        return (selected,)


class FaceLocalRouteSelector:
    """Select one product route without duplicating generation/composite chains.

    R2 and N2 intentionally share the same reference-priority model branch.  N2
    differs in provenance: Image 2 must be generated from a source-derived
    photography brief and explicitly approved by the user before this node will
    allow execution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        prompt_widget = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                "route_mode": (list(ROUTE_MODES), {"default": "N1"}),
                "n1_prompt_en": ("STRING", {"default": DEFAULT_N1_PROMPT_EN, **prompt_widget}),
                "n1_prompt_zh": ("STRING", {"default": DEFAULT_N1_PROMPT_ZH, **prompt_widget}),
                "r1_prompt_en": ("STRING", {"default": DEFAULT_R1_PROMPT_EN, **prompt_widget}),
                "r1_prompt_zh": ("STRING", {"default": DEFAULT_R1_PROMPT_ZH, **prompt_widget}),
                "r2_n2_prompt_en": (
                    "STRING",
                    {"default": DEFAULT_R2_N2_PROMPT_EN, **prompt_widget},
                ),
                "r2_n2_prompt_zh": (
                    "STRING",
                    {"default": DEFAULT_R2_N2_PROMPT_ZH, **prompt_widget},
                ),
                "n2_candidate_user_approved": ("BOOLEAN", {"default": False}),
                "n2_candidate_id": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "selected_prompt_en",
        "selected_prompt_zh",
        "use_identity_reference",
        "reference_priority",
        "preserve_source_eye_interior",
        "route_report_json",
    )
    FUNCTION = "select"
    CATEGORY = "face local edit/0 route"

    def select(
        self,
        route_mode,
        n1_prompt_en,
        n1_prompt_zh,
        r1_prompt_en,
        r1_prompt_zh,
        r2_n2_prompt_en,
        r2_n2_prompt_zh,
        n2_candidate_user_approved,
        n2_candidate_id,
    ):
        route = str(route_mode)
        if route not in ROUTE_MODES:
            raise ValueError(f"Unsupported route mode: {route}")
        prompt_pairs = {
            "N1": (str(n1_prompt_en), str(n1_prompt_zh)),
            "R1": (str(r1_prompt_en), str(r1_prompt_zh)),
            "R2": (str(r2_n2_prompt_en), str(r2_n2_prompt_zh)),
            "N2": (str(r2_n2_prompt_en), str(r2_n2_prompt_zh)),
        }
        prompt_en, prompt_zh = prompt_pairs[route]
        if not prompt_en.strip() or not prompt_zh.strip():
            raise ValueError(f"Route {route} requires non-empty English and Chinese prompts")

        candidate_id = str(n2_candidate_id).strip()
        candidate_approved = bool(n2_candidate_user_approved)
        if route == "N2" and not candidate_approved:
            raise ValueError(
                "N2 is blocked until the user approves a source-derived, angle-matched or front-facing candidate"
            )
        if route == "N2" and not candidate_id:
            raise ValueError("N2 requires a recorded candidate id after user approval")

        use_reference = route in {"R1", "R2", "N2"}
        reference_priority = route in {"R2", "N2"}
        preserve_eye_interior = route == "R1"
        report = {
            "route_mode": route,
            "selected_prompt_en_sha256": hashlib.sha256(prompt_en.encode("utf-8")).hexdigest(),
            "selected_prompt_zh_sha256": hashlib.sha256(prompt_zh.encode("utf-8")).hexdigest(),
            "use_identity_reference": use_reference,
            "reference_priority": reference_priority,
            "preserve_source_eye_interior": preserve_eye_interior,
            "user_reference_required": route in {"R1", "R2"},
            "n2_internal_reference_required": route == "N2",
            "n2_candidate_policy": (
                "source reverse brief -> angle-matched or front-facing candidates -> hard gate -> user selection"
            ),
            "n2_candidate_user_approved": candidate_approved if route == "N2" else None,
            "n2_candidate_id": candidate_id if route == "N2" else None,
            "n2_candidate_is_not_phase1_output": route == "N2",
            "route_switches_only_conditioning_model_and_sigmas": True,
            "generation_chain_shared": True,
            "strict_composite_chain_shared": True,
            "gate_passed": True,
        }
        return (
            prompt_en,
            prompt_zh,
            use_reference,
            reference_priority,
            preserve_eye_interior,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class FaceLocalExactIntegerDownscale:
    """Downscale by an exact integer divisor while preserving 16-pixel alignment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "maximum_short_edge": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 16},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("model_image",)
    FUNCTION = "downscale"
    CATEGORY = "face local edit/3 generation"

    def downscale(self, image, maximum_short_edge):
        source = _image_batch(image)
        source_height, source_width = int(source.shape[1]), int(source.shape[2])
        model_width, model_height, _ = _plan_exact_integer_downscale(
            source_width,
            source_height,
            maximum_short_edge,
        )
        return (_resize_lanczos_exact(source, model_width, model_height),)


class FaceLocalSemanticCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "selection": ("FACE_LOCAL_SELECTION",),
                "final_target_mask": ("MASK",),
                "protection_mask": ("MASK",),
                "edit_region_mode": (
                    list(EDIT_REGION_MODES),
                    {"default": BROAD_HEAD_REGION_MODE},
                ),
                "tile_profile": (list(FACE_TILE_PROFILES), {"default": "大块（高显存）"}),
                "context_left_right": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.20, "max": 1.50, "step": 0.05},
                ),
                "context_top": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.30, "max": 1.50, "step": 0.05},
                ),
                "context_bottom": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.30, "max": 1.50, "step": 0.05},
                ),
                "generation_grow_ratio": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 0.35, "step": 0.01},
                ),
                "composite_grow_ratio": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.0, "max": 0.25, "step": 0.01},
                ),
                "feather_ratio": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.0, "max": 0.10, "step": 0.005},
                ),
                "minimum_safe_margin": (
                    "INT",
                    {"default": 64, "min": 32, "max": 512, "step": 16},
                ),
            },
            "optional": {
                "context_policy": (
                    list(CONTEXT_POLICIES),
                    {"default": SOURCE_EDIT_CONTEXT_POLICY},
                ),
                "writeback_scope_contract_version": (
                    "STRING",
                    {"default": WRITEBACK_SCOPE_CONTRACT_VERSION},
                ),
                "generation_target_megapixels": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "INT", "INT", "INT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = (
        "local_image",
        "local_generation_mask",
        "full_composite_mask",
        "full_composite_alpha",
        "x",
        "y",
        "width",
        "height",
        "crop_preview",
        "report_json",
    )
    FUNCTION = "plan"
    CATEGORY = "face local edit/4 plan"

    def plan(
        self,
        image,
        selection,
        final_target_mask,
        protection_mask,
        edit_region_mode,
        tile_profile,
        context_left_right,
        context_top,
        context_bottom,
        generation_grow_ratio,
        composite_grow_ratio,
        feather_ratio,
        minimum_safe_margin,
        context_policy=SOURCE_EDIT_CONTEXT_POLICY,
        writeback_scope_contract_version=WRITEBACK_SCOPE_CONTRACT_VERSION,
        generation_target_megapixels=0.0,
    ):
        source = _image_batch(image).to(device="cpu")
        full_height, full_width = int(source.shape[1]), int(source.shape[2])
        if selection.get("image_width") != full_width or selection.get("image_height") != full_height:
            raise ValueError("Face selection dimensions do not match the input image")
        face = _selection_face(selection)
        x1, y1, x2, y2 = (float(v) for v in face["bbox_xyxy"])
        face_width, face_height = x2 - x1, y2 - y1
        face_short = min(face_width, face_height)

        mode = str(edit_region_mode)
        if mode not in EDIT_REGION_MODES:
            raise ValueError(f"Unsupported edit_region_mode: {mode}")
        selected_context_policy = str(context_policy)
        if selected_context_policy not in CONTEXT_POLICIES:
            raise ValueError(f"Unsupported context_policy: {selected_context_policy}")
        loaded_writeback_contract = str(writeback_scope_contract_version)
        if loaded_writeback_contract != WRITEBACK_SCOPE_CONTRACT_VERSION:
            raise ValueError(
                "Writeback scope contract mismatch: "
                f"expected {WRITEBACK_SCOPE_CONTRACT_VERSION!r}, "
                f"received {loaded_writeback_contract!r}. Reload ComfyUI through the launcher "
                "before running this Phase-1 workflow."
            )

        target = _mask_batch(final_target_mask, full_height, full_width)[0].numpy() > 0.001
        if not np.any(target):
            raise ValueError("The semantic edit request is empty")
        protection = _mask_batch(protection_mask, full_height, full_width)[0].numpy() > 0.5
        generation_grow_pixels = int(round(face_short * float(generation_grow_ratio)))
        generation_grow_pixels = max(0, min(generation_grow_pixels, 768))
        composite_grow_pixels = int(round(face_short * float(composite_grow_ratio)))
        composite_grow_pixels = max(0, min(composite_grow_pixels, 512))
        protected = _expand_binary_mask(protection, 4)
        feather_pixels = int(round(face_short * float(feather_ratio)))
        feather_pixels = max(0, min(feather_pixels, 256))
        compatibility_pixels = generation_grow_pixels - composite_grow_pixels
        if compatibility_pixels < feather_pixels:
            raise ValueError(
                "Generation grow must leave at least one feather-width compatibility band outside the final composite"
            )
        safe_margin = max(int(minimum_safe_margin), int(math.ceil(face_short * 0.08)))

        requested = target & ~protected
        if not np.any(requested):
            raise ValueError("Protection removes the complete semantic edit request")
        if mode == BROAD_HEAD_REGION_MODE:
            generation_seed = _broad_head_semantic_seed(requested, (x1, y1, x2, y2))
            composite_seed = _broad_head_composite_seed(requested, (x1, y1, x2, y2))
        else:
            generation_seed = requested.copy()
            composite_seed = requested.copy()
        generation_seed &= ~protected
        composite_seed &= ~protected

        # Reuse the verified strict local-edit contract: the semantic shape is
        # expanded twice, once for model generation and once for final writeback.
        # The wider generation support provides a compatibility band outside the
        # final alpha; neither support is ever replaced by the crop rectangle.
        generation_support = _expand_binary_mask(generation_seed, generation_grow_pixels) & ~protected
        composite_support = _expand_binary_mask(composite_seed, composite_grow_pixels) & ~protected
        if not np.any(generation_support):
            raise ValueError("Protection removes the complete expanded generation region")
        if not np.any(composite_support):
            raise ValueError("Protection removes the complete final composite region")
        if np.any(composite_support & ~generation_support):
            raise ValueError("The final composite region must be fully contained by the wider generation region")
        uncovered_request = requested & ~composite_support
        if np.any(uncovered_request):
            raise ValueError("The final composite support does not contain the complete semantic request")

        final_writeback_scope_gate_applied = mode == BROAD_HEAD_REGION_MODE
        automatic_upper_neck_writeback_limit_y = None
        final_composite_pixels_below_upper_neck_limit = 0
        final_composite_y, _ = np.nonzero(composite_support)
        final_composite_bottom_y = int(final_composite_y.max())
        if final_writeback_scope_gate_applied:
            # The limit is inclusive. Generation may extend farther down for
            # contextual synthesis, but no final alpha/support pixel may cross
            # this face-relative upper-neck boundary into shoulder/chest/body.
            automatic_upper_neck_writeback_limit_y = min(
                full_height - 1,
                int(
                    math.ceil(
                        y2
                        + face_height * BROAD_COMPOSITE_NECK_BOTTOM_RATIO
                        + composite_grow_pixels
                    )
                ),
            )
            final_composite_pixels_below_upper_neck_limit = int(
                np.count_nonzero(
                    composite_support[automatic_upper_neck_writeback_limit_y + 1 :]
                )
            )
        final_writeback_scope_gate_passed = bool(
            not final_writeback_scope_gate_applied
            or final_composite_pixels_below_upper_neck_limit == 0
        )
        if not final_writeback_scope_gate_passed:
            raise ValueError(
                "The Phase-1 final writeback extends below the permitted head/upper-neck boundary "
                "into shoulder, chest, or body pixels. Keep the wider area as generation context only; "
                "erase the out-of-scope manual addition or move it into protection."
            )

        generation_y, generation_x = np.nonzero(generation_support)
        edit_margin_bounds = {
            "left": float(generation_x.min() - safe_margin),
            "right": float(generation_x.max() + 1 + safe_margin),
            "top": float(generation_y.min() - safe_margin),
            "bottom": float(generation_y.max() + 1 + safe_margin),
        }

        context_requested = {
            "left": float(context_left_right) * face_width,
            "right": float(context_left_right) * face_width,
            "top": float(context_top) * face_height,
            "bottom": float(context_bottom) * face_height,
        }
        context_available = {
            "left": max(0.0, x1),
            "right": max(0.0, full_width - x2),
            "top": max(0.0, y1),
            "bottom": max(0.0, full_height - y2),
        }
        context_retention = {
            key: min(1.0, context_available[key] / requested) if requested > 0 else 1.0
            for key, requested in context_requested.items()
        }
        context_retention_gate_applied = selected_context_policy == SOURCE_EDIT_CONTEXT_POLICY
        if context_retention_gate_applied and min(context_retention.values()) < MIN_CONTEXT_RETENTION_RATIO:
            raise ValueError(
                "The image does not contain enough hair/ear/neck/light context around the selected face; "
                f"retention={context_retention}, required={MIN_CONTEXT_RETENTION_RATIO}"
            )

        required_x0 = max(
            0.0,
            min(x1 - context_requested["left"], edit_margin_bounds["left"]),
        )
        required_x1 = min(
            float(full_width),
            max(x2 + context_requested["right"], edit_margin_bounds["right"]),
        )
        required_y0 = max(
            0.0,
            min(y1 - context_requested["top"], edit_margin_bounds["top"]),
        )
        required_y1 = min(
            float(full_height),
            max(y2 + context_requested["bottom"], edit_margin_bounds["bottom"]),
        )

        allow_source_edge_trim = selected_context_policy == SOURCE_EDGE_COMPAT_CONTEXT_POLICY
        crop_x0, crop_x1 = _aligned_axis(
            required_x0,
            required_x1,
            full_width,
            allow_source_edge_trim=allow_source_edge_trim,
        )
        crop_y0, crop_y1 = _aligned_axis(
            required_y0,
            required_y1,
            full_height,
            allow_source_edge_trim=allow_source_edge_trim,
        )
        crop_width, crop_height = crop_x1 - crop_x0, crop_y1 - crop_y0
        native_crop_pixels = crop_width * crop_height
        target_megapixels = max(0.0, float(generation_target_megapixels))
        profile_width, profile_height = crop_width, crop_height
        if target_megapixels > 0.0:
            target_pixels = max(MULTIPLE * MULTIPLE, int(round(target_megapixels * 1_000_000)))
            if native_crop_pixels > target_pixels:
                scale = math.sqrt(target_pixels / native_crop_pixels)
                profile_width = max(MULTIPLE, int(round(crop_width * scale / MULTIPLE)) * MULTIPLE)
                profile_height = max(MULTIPLE, int(round(crop_height * scale / MULTIPLE)) * MULTIPLE)
        max_long, max_short, max_pixels = FACE_TILE_PROFILES[str(tile_profile)]
        if (
            max(profile_width, profile_height) > max_long
            or min(profile_width, profile_height) > max_short
            or profile_width * profile_height > max_pixels
        ):
            raise ValueError(
                f"Planned generation crop {profile_width}x{profile_height} "
                f"(native source crop {crop_width}x{crop_height}) exceeds profile {tile_profile}; "
                "choose a larger profile or a source with a smaller face. Splitting through facial features is forbidden."
            )

        generation_window_inset = None
        composite_window_inset = None

        local_image = source[:, crop_y0:crop_y1, crop_x0:crop_x1, :].clone()
        full_generation_mask = torch.from_numpy(
            generation_support.astype(np.float32, copy=False)
        ).unsqueeze(0)
        local_mask = full_generation_mask[:, crop_y0:crop_y1, crop_x0:crop_x1].clone()
        full_composite_mask = torch.from_numpy(
            composite_support.astype(np.float32, copy=False)
        ).unsqueeze(0)
        blur_map = np.where(composite_support, feather_pixels, 0).astype(np.float32, copy=False)
        full_alpha_np = _inward_feather_alpha(composite_support, blur_map)
        if not np.all(full_alpha_np[~composite_support] == 0.0):
            raise RuntimeError("Final alpha escaped the strict composite support")
        full_alpha = torch.from_numpy(full_alpha_np).unsqueeze(0)

        preview = _overlay_mask(source, full_generation_mask, (1.0, 0.65, 0.0))
        preview = _overlay_mask(preview, full_composite_mask, (1.0, 0.05, 0.05))
        preview_np = (preview[0].numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        preview_image = Image.fromarray(preview_np)
        draw = ImageDraw.Draw(preview_image)
        draw.rectangle((crop_x0, crop_y0, crop_x1 - 1, crop_y1 - 1), outline=(255, 220, 0), width=4)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 80), width=3)
        preview = torch.from_numpy(np.asarray(preview_image).copy()).float().div(255.0).unsqueeze(0)

        generation_y, generation_x = np.nonzero(generation_support)
        distances = {
            "left": int(generation_x.min() - crop_x0),
            "right": int(crop_x1 - 1 - generation_x.max()),
            "top": int(generation_y.min() - crop_y0),
            "bottom": int(crop_y1 - 1 - generation_y.max()),
        }
        source_edge_exemptions = {
            # A crop boundary that is also the source-image boundary has no
            # omitted source pixels beyond it.  The separate context-retention
            # gate already decides whether enough real hair/ear/neck/light
            # context exists; requiring the editable support itself to touch
            # the source edge incorrectly rejects valid near-edge faces.
            "left": bool(crop_x0 == 0),
            "right": bool(crop_x1 == full_width),
            "top": bool(crop_y0 == 0),
            "bottom": bool(crop_y1 == full_height),
        }
        safe_margin_gate_by_side = {
            key: bool(distances[key] >= safe_margin or source_edge_exemptions[key])
            for key in distances
        }
        if not all(safe_margin_gate_by_side.values()):
            raise RuntimeError(
                "Semantic crop safe margin failed: "
                f"actual={distances}, source_edge_exemptions={source_edge_exemptions}, "
                f"required={safe_margin}"
            )
        report = {
            "single_semantic_face_tile": True,
            "internal_seam_pixels": 0,
            "crop_xywh": [crop_x0, crop_y0, crop_width, crop_height],
            "crop_multiple": MULTIPLE,
            "face_bbox_xyxy": [x1, y1, x2, y2],
            "face_short_side": face_short,
            "edit_region_mode": mode,
            "broad_prompt_driven_head_region": mode == BROAD_HEAD_REGION_MODE,
            "automatic_face_oval_is_semantic_seed_only": mode == BROAD_HEAD_REGION_MODE,
            "composite_boundary_is_not_anatomical_face_boundary": mode
            == BROAD_HEAD_REGION_MODE,
            "support_shape": "expanded_semantic_head_envelope"
            if mode == BROAD_HEAD_REGION_MODE
            else "expanded_face_oval",
            "rectangular_composite_forbidden": True,
            "verified_global_inward_feather_reused": True,
            "mask_roles_separated": True,
            "generation_and_composite_seeds_are_independent": mode
            == BROAD_HEAD_REGION_MODE,
            "automatic_composite_excludes_shoulders_and_chest": bool(
                final_writeback_scope_gate_applied
                and final_writeback_scope_gate_passed
            ),
            "final_writeback_scope_gate_applied": final_writeback_scope_gate_applied,
            "writeback_scope_contract_version": WRITEBACK_SCOPE_CONTRACT_VERSION,
            "automatic_upper_neck_writeback_limit_y": automatic_upper_neck_writeback_limit_y,
            "final_composite_bottom_y": final_composite_bottom_y,
            "final_composite_pixels_below_upper_neck_limit": (
                final_composite_pixels_below_upper_neck_limit
            ),
            "final_writeback_scope_gate_passed": final_writeback_scope_gate_passed,
            "generation_grow_pixels": generation_grow_pixels,
            "final_composite_grow_pixels": composite_grow_pixels,
            "compatibility_band_width_pixels": compatibility_pixels,
            "final_inward_feather_pixels": feather_pixels,
            "safe_margin_required": safe_margin,
            "safe_margin_actual": distances,
            "source_edge_safe_margin_exemptions": source_edge_exemptions,
            "safe_margin_gate_by_side": safe_margin_gate_by_side,
            "context_policy": selected_context_policy,
            "context_retention_gate_applied": context_retention_gate_applied,
            "source_edge_context_accepted": bool(
                selected_context_policy == SOURCE_EDGE_COMPAT_CONTEXT_POLICY
                and min(context_retention.values()) < MIN_CONTEXT_RETENTION_RATIO
            ),
            "generation_window_inset_pixels": generation_window_inset,
            "final_composite_window_inset_pixels": composite_window_inset,
            "context_retention_required": (
                MIN_CONTEXT_RETENTION_RATIO if context_retention_gate_applied else 0.0
            ),
            "context_requested_pixels": {
                key: round(value, 3) for key, value in context_requested.items()
            },
            "context_available_pixels": {
                key: round(value, 3) for key, value in context_available.items()
            },
            "context_retention_actual": {
                key: round(value, 6) for key, value in context_retention.items()
            },
            "context_clipped_by_source_edge": {
                key: context_available[key] < context_requested[key]
                for key in context_requested
            },
            "generation_pixels": int(np.count_nonzero(generation_support)),
            "final_composite_pixels": int(np.count_nonzero(composite_support)),
            "confirmed_protection_pixels": int(np.count_nonzero(protection)),
            "expanded_protection_pixels": int(np.count_nonzero(protected)),
            "protection_excluded_from_generation": not bool(
                np.any(generation_support & protected)
            ),
            "protection_excluded_from_final_composite": not bool(
                np.any(composite_support & protected)
            ),
            "semantic_request_pixels": int(np.count_nonzero(requested)),
            "semantic_head_seed_pixels": int(np.count_nonzero(generation_seed)),
            "final_composite_seed_pixels": int(np.count_nonzero(composite_seed)),
            "semantic_request_inside_final_composite": not bool(
                np.any(requested & ~composite_support)
            ),
            "compatibility_band_pixels": int(
                np.count_nonzero(generation_support & ~composite_support)
            ),
            "generation_contains_final_composite": not bool(
                np.any(composite_support & ~generation_support)
            ),
            "outside_final_composite_alpha_is_exact_zero": True,
            "profile": {
                "name": str(tile_profile),
                "max_long_side": max_long,
                "max_short_side": max_short,
                "max_pixels": max_pixels,
                "native_crop_width": crop_width,
                "native_crop_height": crop_height,
                "generation_target_megapixels": target_megapixels,
                "planned_generation_width": profile_width,
                "planned_generation_height": profile_height,
            },
            "gate_passed": True,
        }
        return (
            local_image,
            local_mask,
            full_composite_mask,
            full_alpha,
            crop_x0,
            crop_y0,
            crop_width,
            crop_height,
            preview,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


_BOUNDARY_PERCENTILE_SAMPLE_LIMIT = 262_144


def _bounded_even_sample(values: torch.Tensor, limit: int) -> torch.Tensor:
    """Return a deterministic, evenly distributed 1-D sample.

    ``torch.quantile`` rejects very large tensors on some PyTorch builds.  The
    boundary report is an auxiliary diagnostic, so exact counts, means, and
    maxima are accumulated separately while only percentile estimation is
    bounded.  Even spacing is deterministic and covers the entire flattened
    boundary instead of biasing the report toward its first pixels.
    """

    flat = values.reshape(-1)
    count = int(flat.numel())
    if count <= limit:
        return flat.float()
    indices = torch.linspace(
        0,
        count - 1,
        steps=limit,
        device=flat.device,
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    return flat.index_select(0, indices).float()


def _boundary_continuity_metrics(
    original: torch.Tensor,
    result: torch.Tensor,
    support: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, Any]:
    """Measure the actual inside/outside pixel crossings of the final writeback boundary.

    Exact equality outside the mask proves containment, but it cannot prove that
    the first generated pixels inside the mask meet the source naturally.  This
    diagnostic therefore measures every horizontal and vertical inside/outside
    pair, reports the alpha actually applied to the inner endpoint, and compares
    the final RGB gradient with the original gradient at the same pair.
    """

    crossing_pairs = 0
    original_sum = 0.0
    final_sum = 0.0
    excess_sum = 0.0
    alpha_sum = 0.0
    excess_max = 0.0
    alpha_max = 0.0
    excess_percentile_samples: list[torch.Tensor] = []
    alpha_percentile_samples: list[torch.Tensor] = []

    pair_slices = (
        (
            original[:, :, :-1, :],
            original[:, :, 1:, :],
            result[:, :, :-1, :],
            result[:, :, 1:, :],
            support[:, :, :-1],
            support[:, :, 1:],
            alpha[:, :, :-1],
            alpha[:, :, 1:],
        ),
        (
            original[:, :-1, :, :],
            original[:, 1:, :, :],
            result[:, :-1, :, :],
            result[:, 1:, :, :],
            support[:, :-1, :],
            support[:, 1:, :],
            alpha[:, :-1, :],
            alpha[:, 1:, :],
        ),
    )
    for original_a, original_b, result_a, result_b, support_a, support_b, alpha_a, alpha_b in pair_slices:
        crossing = torch.logical_xor(support_a, support_b)
        if not torch.any(crossing):
            continue
        original_gradient = torch.mean(torch.abs(original_a - original_b), dim=-1)[crossing]
        final_gradient = torch.mean(torch.abs(result_a - result_b), dim=-1)[crossing]
        excess_gradient = torch.clamp(final_gradient - original_gradient, min=0.0)
        inner_alpha = torch.where(support_a, alpha_a, alpha_b)[crossing]
        pair_count = int(inner_alpha.numel())

        crossing_pairs += pair_count
        original_sum += float(original_gradient.double().sum().item())
        final_sum += float(final_gradient.double().sum().item())
        excess_sum += float(excess_gradient.double().sum().item())
        alpha_sum += float(inner_alpha.double().sum().item())
        excess_max = max(excess_max, float(excess_gradient.max().item()))
        alpha_max = max(alpha_max, float(inner_alpha.max().item()))
        excess_percentile_samples.append(
            _bounded_even_sample(excess_gradient, _BOUNDARY_PERCENTILE_SAMPLE_LIMIT)
        )
        alpha_percentile_samples.append(
            _bounded_even_sample(inner_alpha, _BOUNDARY_PERCENTILE_SAMPLE_LIMIT)
        )

    if crossing_pairs == 0:
        return {
            "boundary_crossing_pairs": 0,
            "boundary_percentile_method": "exact",
            "boundary_percentile_sample_pairs": 0,
            "boundary_inner_alpha_mean": 0.0,
            "boundary_inner_alpha_p95": 0.0,
            "boundary_inner_alpha_max": 0.0,
            "boundary_original_rgb_gradient_mean": 0.0,
            "boundary_final_rgb_gradient_mean": 0.0,
            "introduced_boundary_rgb_gradient_excess_mean": 0.0,
            "introduced_boundary_rgb_gradient_excess_p95": 0.0,
            "introduced_boundary_rgb_gradient_excess_max": 0.0,
            "feather_boundary_alpha_max_allowed": 0.10,
            "feather_boundary_gate_passed": True,
            "automatic_metric_is_auxiliary": True,
            "skin_tone_light_texture_continuity_requires_visual_review": True,
        }

    excess_sample = torch.cat(excess_percentile_samples)
    alpha_sample = torch.cat(alpha_percentile_samples)
    percentile_sample_pairs = int(alpha_sample.numel())
    percentile_method = (
        "exact"
        if percentile_sample_pairs == crossing_pairs
        else "deterministic_even_sample"
    )
    alpha_max_allowed = 0.10

    def percentile(values: torch.Tensor, quantile: float) -> float:
        return float(torch.quantile(values.float(), quantile).item())

    return {
        "boundary_crossing_pairs": crossing_pairs,
        "boundary_percentile_method": percentile_method,
        "boundary_percentile_sample_pairs": percentile_sample_pairs,
        "boundary_inner_alpha_mean": alpha_sum / crossing_pairs,
        "boundary_inner_alpha_p95": percentile(alpha_sample, 0.95),
        "boundary_inner_alpha_max": alpha_max,
        "boundary_original_rgb_gradient_mean": original_sum / crossing_pairs,
        "boundary_final_rgb_gradient_mean": final_sum / crossing_pairs,
        "introduced_boundary_rgb_gradient_excess_mean": excess_sum / crossing_pairs,
        "introduced_boundary_rgb_gradient_excess_p95": percentile(excess_sample, 0.95),
        "introduced_boundary_rgb_gradient_excess_max": excess_max,
        "feather_boundary_alpha_max_allowed": alpha_max_allowed,
        "feather_boundary_gate_passed": bool(alpha_max <= alpha_max_allowed + 1.0e-6),
        "automatic_metric_is_auxiliary": True,
        "skin_tone_light_texture_continuity_requires_visual_review": True,
    }


class FaceLocalContextColorHarmonize:
    """Match a generated face tile to source context without masked-zero bias.

    The generation mask is an exclusion mask for statistics, not a compositing
    mask.  Only pixels outside it contribute to the Lab transfer.  The corrected
    tile may then be passed to ``FaceLocalStrictComposite``, whose independent
    short writeback alpha remains authoritative.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE",),
                "reference": ("IMAGE",),
                "generation_mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "minimum_context_pixels": (
                    "INT",
                    {
                        "default": COLOR_HARMONIZATION_MINIMUM_CONTEXT_PIXELS,
                        "min": 1,
                        "max": MAX_RESOLUTION * MAX_RESOLUTION,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "color_harmonization_contract_version": (
                    "STRING",
                    {"default": COLOR_HARMONIZATION_CONTRACT_VERSION},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("harmonized_image", "report_json")
    FUNCTION = "harmonize"
    CATEGORY = "face local edit/4 continuity"

    def harmonize(
        self,
        target,
        reference,
        generation_mask,
        strength,
        minimum_context_pixels,
        color_harmonization_contract_version=COLOR_HARMONIZATION_CONTRACT_VERSION,
    ):
        loaded_contract = str(color_harmonization_contract_version)
        if loaded_contract != COLOR_HARMONIZATION_CONTRACT_VERSION:
            raise ValueError(
                "Color-harmonization runtime contract mismatch: "
                f"expected {COLOR_HARMONIZATION_CONTRACT_VERSION!r}, received {loaded_contract!r}"
            )

        generated = _image_batch(target).to(dtype=torch.float32)
        source = _image_batch(reference).to(device=generated.device, dtype=torch.float32)
        if generated.shape != source.shape:
            raise ValueError(
                "Target and reference must have the same BHWC shape, received "
                f"{tuple(generated.shape)} and {tuple(source.shape)}"
            )
        height, width = int(generated.shape[1]), int(generated.shape[2])
        excluded = _mask_batch(generation_mask, height, width).to(generated.device) >= 0.5
        valid_context = ~excluded
        context_pixels = int(torch.count_nonzero(valid_context).item())
        minimum_pixels = int(minimum_context_pixels)
        if context_pixels < minimum_pixels:
            raise ValueError(
                "Insufficient ungenerated context for reliable color harmonization: "
                f"{context_pixels} valid pixels, minimum {minimum_pixels}"
            )

        generated_lab = _srgb_to_lab(generated)
        source_lab = _srgb_to_lab(source)
        generated_std, generated_mean, _ = _exact_masked_channel_stats(
            generated_lab, valid_context
        )
        source_std, source_mean, _ = _exact_masked_channel_stats(source_lab, valid_context)

        # Retain the mature upstream safeguard for information-poor channels,
        # but compute every statistic over the true valid set rather than a
        # zero-padded flattened tensor.  A global affine transfer cannot recover
        # variation from a uniform channel, so both sides must carry signal.
        active_channels = (generated_std >= 1.0) & (source_std >= 1.0)
        safe_generated_std = generated_std.clamp_min(1.0e-6)
        gain = torch.where(active_channels, source_std / safe_generated_std, torch.ones_like(source_std))
        corrected_lab = torch.where(
            active_channels,
            (generated_lab - generated_mean) * gain + source_mean,
            generated_lab,
        )
        corrected_rgb = _lab_to_srgb(corrected_lab).clamp(0.0, 1.0)
        amount = float(strength)
        harmonized = ((1.0 - amount) * generated + amount * corrected_rgb).clamp(0.0, 1.0)
        harmonized_lab = _srgb_to_lab(harmonized)
        harmonized_std, harmonized_mean, _ = _exact_masked_channel_stats(
            harmonized_lab, valid_context
        )

        def triplet(value: torch.Tensor) -> list[float]:
            return [float(item) for item in value[0, :, 0, 0].detach().cpu().tolist()]

        channel_names = ("L", "a", "b")
        active_flags = [bool(item) for item in active_channels[0, :, 0, 0].detach().cpu().tolist()]
        before_mean_delta = torch.abs(generated_mean - source_mean)
        after_mean_delta = torch.abs(harmonized_mean - source_mean)
        before_std_delta = torch.abs(generated_std - source_std)
        after_std_delta = torch.abs(harmonized_std - source_std)
        total_pixels = height * width
        report = {
            "algorithm": "exact-valid-context-lab-mean-std-transfer",
            "color_harmonization_contract_version": COLOR_HARMONIZATION_CONTRACT_VERSION,
            "statistics_region": "generation_mask_below_0.5_only",
            "excluded_pixels_are_zero_padded_into_statistics": False,
            "target_reference_shape_equal": True,
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "valid_context_pixels": context_pixels,
            "excluded_generation_pixels": total_pixels - context_pixels,
            "valid_context_ratio": context_pixels / float(total_pixels),
            "minimum_context_pixels": minimum_pixels,
            "strength": amount,
            "channel_order": list(channel_names),
            "active_channels": {
                name: enabled for name, enabled in zip(channel_names, active_flags)
            },
            "generated_context_mean_lab_before": triplet(generated_mean),
            "source_context_mean_lab": triplet(source_mean),
            "harmonized_context_mean_lab_after": triplet(harmonized_mean),
            "generated_context_std_lab_before": triplet(generated_std),
            "source_context_std_lab": triplet(source_std),
            "harmonized_context_std_lab_after": triplet(harmonized_std),
            "applied_gain_lab": triplet(gain),
            "mean_absolute_stat_delta_before": float(before_mean_delta.mean().item()),
            "mean_absolute_stat_delta_after": float(after_mean_delta.mean().item()),
            "std_absolute_stat_delta_before": float(before_std_delta.mean().item()),
            "std_absolute_stat_delta_after": float(after_std_delta.mean().item()),
            "gate_passed": True,
            "visual_skin_tone_lighting_texture_review_still_required": True,
        }
        return harmonized, json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalStrictComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "full_composite_mask": ("MASK",),
                "full_composite_alpha": ("MASK",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
            },
            "optional": {
                "local_composite_alpha": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("final_image", "strict_composite_mask", "difference_preview", "report_json")
    FUNCTION = "composite"
    CATEGORY = "face local edit/5 strict merge"

    def composite(
        self,
        original,
        generated_local,
        full_composite_mask,
        full_composite_alpha,
        x,
        y,
        width,
        height,
        local_composite_alpha=None,
    ):
        base = _image_batch(original).to(device="cpu", dtype=torch.float32)
        candidate = _image_batch(generated_local).to(device="cpu", dtype=torch.float32)
        full_height, full_width = int(base.shape[1]), int(base.shape[2])
        px, py, tile_width, tile_height = int(x), int(y), int(width), int(height)
        if candidate.shape[1:3] != (tile_height, tile_width):
            raise ValueError(
                f"Generated local shape {tuple(candidate.shape[1:3])} does not match planned {(tile_height, tile_width)}"
            )
        (
            original_x0,
            original_y0,
            original_x1,
            original_y1,
            local_x0,
            local_y0,
            local_x1,
            local_y1,
        ) = _crop_intersection(
            px,
            py,
            tile_width,
            tile_height,
            full_width,
            full_height,
        )
        support = _mask_batch(full_composite_mask, full_height, full_width) > 0.5
        alpha = _mask_batch(full_composite_alpha, full_height, full_width)
        alpha_source = "semantic_crop_full_composite_alpha"
        alpha_quantization_floor = 0.0
        if local_composite_alpha is not None:
            local_override = _mask_batch(
                local_composite_alpha, tile_height, tile_width
            )
            # A mask written through an 8-bit image cannot represent values at
            # or below half of one byte step.  Remove only those sub-byte
            # feather tails before deriving the strict support; otherwise the
            # support threshold and the exact outside-support check contradict
            # one another.
            alpha_quantization_floor = 0.5 / 255.0
            local_override = torch.where(
                local_override > alpha_quantization_floor,
                local_override,
                torch.zeros_like(local_override),
            )
            alpha = torch.zeros(
                (1, full_height, full_width), dtype=torch.float32
            )
            alpha[
                :,
                original_y0:original_y1,
                original_x0:original_x1,
            ] = local_override[
                :,
                local_y0:local_y1,
                local_x0:local_x1,
            ]
            support = alpha > 0.0
            alpha_source = (
                "internal_adaptive_difference_union_manual_correction_local_alpha"
            )
        strict_mask_pixels = int(torch.count_nonzero(support).item())
        if strict_mask_pixels == 0:
            raise ValueError("Strict composite mask is empty; refusing to return the unchanged source image")
        if torch.any(alpha[~support] != 0):
            raise ValueError("Composite alpha is non-zero outside the strict generation mask")
        effective_alpha_pixels = int(torch.count_nonzero(alpha[support] > 0).item())
        if effective_alpha_pixels == 0:
            raise ValueError("Strict composite alpha is zero inside the mask; refusing a no-op composite")
        result = base.clone()
        local_support = support[
            :,
            original_y0:original_y1,
            original_x0:original_x1,
        ]
        local_alpha = alpha[
            :,
            original_y0:original_y1,
            original_x0:original_x1,
        ].unsqueeze(-1)
        destination = result[
            :,
            original_y0:original_y1,
            original_x0:original_x1,
            :,
        ]
        candidate_intersection = candidate[
            :,
            local_y0:local_y1,
            local_x0:local_x1,
            :,
        ]
        blended = (
            destination * (1.0 - local_alpha)
            + candidate_intersection * local_alpha
        )
        destination[local_support] = blended[local_support]
        outside_equal = torch.equal(result[~support], base[~support])
        if not outside_equal:
            raise RuntimeError("Strict composite changed pixels outside the generation mask")
        difference = torch.abs(result - base)
        boundary = _boundary_continuity_metrics(base, result, support, alpha)
        report = {
            "input_width": full_width,
            "input_height": full_height,
            "output_width": int(result.shape[2]),
            "output_height": int(result.shape[1]),
            "dimensions_equal": result.shape == base.shape,
            "strict_mask_pixels": strict_mask_pixels,
            "effective_alpha_pixels": effective_alpha_pixels,
            "outside_mask_mismatch_pixels_float": 0,
            "outside_mask_max_abs_diff_float": 0.0,
            "outside_mask_is_exact_original": outside_equal,
            "internal_tile_count": 1,
            "internal_seam_pixels": 0,
            "boundary_continuity": boundary,
            "crop_xywh": [px, py, tile_width, tile_height],
            "original_intersection_xyxy": [
                original_x0,
                original_y0,
                original_x1,
                original_y1,
            ],
            "local_intersection_xyxy": [
                local_x0,
                local_y0,
                local_x1,
                local_y1,
            ],
            "padded_context_pixels": int(
                tile_width * tile_height
                - (original_x1 - original_x0) * (original_y1 - original_y0)
            ),
            "alpha_source": alpha_source,
            "alpha_quantization_floor": alpha_quantization_floor,
            "gate_passed": bool(
                outside_equal
                and result.shape == base.shape
                and boundary["feather_boundary_gate_passed"]
            ),
        }
        return result.clamp(0.0, 1.0), support.float(), difference.clamp(0.0, 1.0), json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalDetailHarmonizer:
    """Reduce excess untouched-region microdetail without softening the composite.

    The final composite can contain a face reconstructed from a smaller model
    canvas while the untouched body and background retain native-camera
    microcontrast.  This node attenuates only the high-frequency residual
    outside the final composite alpha.  Fractional alpha values preserve the
    same transition ramp used by the strict composite, while strong scene
    contours receive an independent protection weight so hair, clothing and
    object boundaries do not turn into soft halos.

    The first input and first output are both IMAGE deliberately: normal
    ComfyUI bypass mode therefore returns the input image byte-for-byte.
    """

    EDGE_PROTECTION_START = 0.035
    EDGE_PROTECTION_END = 0.120

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "protected_mask": ("MASK",),
                "enabled": ("BOOLEAN", {"default": False}),
                "detail_radius": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05},
                ),
                "outside_detail_retention": (
                    "FLOAT",
                    {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "strong_edge_protection": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = (
        "detail_harmonized_image",
        "protected_composite_region",
        "report_json",
    )
    FUNCTION = "harmonize"
    CATEGORY = "face local edit/8 final output"

    def harmonize(
        self,
        image,
        protected_mask,
        enabled=False,
        detail_radius=1.0,
        outside_detail_retention=0.80,
        strong_edge_protection=0.90,
    ):
        if not bool(enabled):
            if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
                raise ValueError("IMAGE must be an HWC or BHWC torch tensor")
            if image.ndim == 3:
                batch, height, width = 1, int(image.shape[0]), int(image.shape[1])
            else:
                batch, height, width = (
                    int(image.shape[0]),
                    int(image.shape[1]),
                    int(image.shape[2]),
                )
            protection = torch.zeros(
                (batch, height, width),
                device=image.device,
                dtype=torch.float32,
            )
            report = {
                "algorithm": "final-composite-mask-protected-exterior-high-frequency-attenuation-v2",
                "processing_applied": False,
                "reason": "disabled",
                "input_returned_without_conversion": True,
                "image_dimensions": [width, height],
            }
            return (
                image,
                protection,
                json.dumps(report, ensure_ascii=False, indent=2),
            )

        source = _image_batch(image).to(device="cpu", dtype=torch.float32)
        height, width = int(source.shape[1]), int(source.shape[2])
        radius = float(detail_radius)
        retention = float(outside_detail_retention)
        edge_protection_strength = float(strong_edge_protection)
        if not 0.25 <= radius <= 4.0:
            raise ValueError("detail_radius must be between 0.25 and 4.0")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("outside_detail_retention must be between 0 and 1")
        if not 0.0 <= edge_protection_strength <= 1.0:
            raise ValueError("strong_edge_protection must be between 0 and 1")

        protection = _mask_batch(protected_mask, height, width)
        if not torch.isfinite(protection).all():
            raise ValueError("protected_mask contains non-finite values")
        protection_np = protection[0].numpy()

        # Retention 1.0 is a useful neutral setting and must be a mathematically
        # exact pass-through, not a blur followed by an approximate rebuild.
        if retention == 1.0:
            report = {
                "algorithm": "final-composite-mask-protected-exterior-high-frequency-attenuation-v2",
                "processing_applied": False,
                "reason": "outside_detail_retention_is_one",
                "detail_radius": radius,
                "outside_detail_retention": retention,
                "strong_edge_protection": edge_protection_strength,
                "composite_core_exact": True,
                "image_dimensions": [width, height],
            }
            return (
                source.clone(),
                protection,
                json.dumps(report, ensure_ascii=False, indent=2),
            )

        source_np = source.numpy()
        low_np = ndimage.gaussian_filter(
            source_np,
            sigma=(0.0, radius, radius, 0.0),
            mode="reflect",
        ).astype(np.float32, copy=False)
        high_np = source_np - low_np

        high_magnitude = np.max(np.abs(high_np), axis=-1)
        edge_weight = _smoothstep_numpy(
            self.EDGE_PROTECTION_START,
            self.EDGE_PROTECTION_END,
            high_magnitude,
        )
        outside_gain = retention + (
            (1.0 - retention) * edge_weight * edge_protection_strength
        )
        gain = protection_np + (1.0 - protection_np) * outside_gain
        result_np = low_np + high_np * gain[..., None]

        # The fully composited core is copied back exactly.  This prevents the
        # harmonizer itself from making the AI-generated region any softer.
        exact_composite_core = protection_np >= (1.0 - 1.0e-7)
        result_np[:, exact_composite_core, :] = source_np[:, exact_composite_core, :]
        result = torch.from_numpy(
            np.clip(result_np, 0.0, 1.0).astype(np.float32, copy=False)
        )

        outside = protection_np <= 1.0e-6
        high_before = float(np.mean(np.abs(high_np[:, outside, :]))) if np.any(outside) else 0.0
        result_low_np = ndimage.gaussian_filter(
            result_np,
            sigma=(0.0, radius, radius, 0.0),
            mode="reflect",
        ).astype(np.float32, copy=False)
        high_after = (
            float(np.mean(np.abs((result_np - result_low_np)[:, outside, :])))
            if np.any(outside)
            else 0.0
        )
        composite_core_exact = bool(
            np.array_equal(
                result_np[:, exact_composite_core, :],
                source_np[:, exact_composite_core, :],
            )
        )
        report = {
            "algorithm": "final-composite-mask-protected-exterior-high-frequency-attenuation-v2",
            "processing_applied": True,
            "formula": "low + high * composite_alpha_or_edge_aware_gain",
            "detail_radius": radius,
            "outside_detail_retention": retention,
            "strong_edge_protection": edge_protection_strength,
            "edge_protection_thresholds": [
                self.EDGE_PROTECTION_START,
                self.EDGE_PROTECTION_END,
            ],
            "protection_source": "final_strict_composite_alpha",
            "processing_region": "inverse_of_final_composite_alpha",
            "composite_core_exact": composite_core_exact,
            "outside_mean_absolute_high_frequency_before": high_before,
            "outside_mean_absolute_high_frequency_after": high_after,
            "image_dimensions": [width, height],
            "visual_review_still_required": True,
        }
        return result, protection, json.dumps(report, ensure_ascii=False, indent=2)


def _landmark_metrics(face: dict[str, Any]) -> dict[str, dict[str, float]]:
    points = np.asarray(face["landmarks_xy"], dtype=np.float64)
    if points.shape[0] < 478:
        raise ValueError(f"Expected at least 478 MediaPipe landmarks, received {points.shape[0]}")

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(points[a] - points[b]))

    interocular = max(1.0e-6, distance(33, 263))
    face_height = max(1.0e-6, distance(10, 152))
    return {
        "eyes": {
            "left_width": distance(33, 133) / interocular,
            "right_width": distance(362, 263) / interocular,
            "eye_spacing": distance(133, 362) / interocular,
        },
        "nose": {
            "nose_width": distance(98, 327) / interocular,
            "nose_length": distance(168, 2) / face_height,
        },
        "mouth": {
            "mouth_width": distance(61, 291) / interocular,
            "lip_opening": distance(13, 14) / interocular,
        },
        "jaw": {
            "cheek_width": distance(234, 454) / face_height,
            "jaw_width": distance(172, 397) / face_height,
        },
        "brows": {
            "left_brow_eye": distance(105, 159) / interocular,
            "right_brow_eye": distance(334, 386) / interocular,
        },
    }


def _face_geometry(face: dict[str, Any]) -> dict[str, Any]:
    bbox = np.asarray(face["bbox_xyxy"], dtype=np.float64)
    if bbox.shape != (4,):
        raise ValueError(f"Expected bbox_xyxy with four values, received shape {bbox.shape}")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(1.0e-6, x2 - x1)
    height = max(1.0e-6, y2 - y1)
    points = np.asarray(face["landmarks_xy"], dtype=np.float64)
    if points.shape[0] < 478:
        raise ValueError(f"Expected at least 478 MediaPipe landmarks, received {points.shape[0]}")
    left_eye_center = (points[33] + points[133]) * 0.5
    right_eye_center = (points[362] + points[263]) * 0.5
    eye_axis = right_eye_center - left_eye_center
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "width": width,
        "height": height,
        "area": width * height,
        "center_x": (x1 + x2) * 0.5,
        "center_y": (y1 + y2) * 0.5,
        "aspect_ratio": width / height,
        "eye_axis_roll_degrees": float(math.degrees(math.atan2(eye_axis[1], eye_axis[0]))),
    }


def _angle_distance_degrees(first: float, second: float) -> float:
    return abs((float(second) - float(first) + 180.0) % 360.0 - 180.0)


def _normal_photo_geometry_gate(
    source_faces: list[dict[str, Any]],
    result_faces: list[dict[str, Any]],
    source_index: int,
    result_index: int,
) -> dict[str, Any]:
    thresholds = {
        "face_width_ratio_min": 0.75,
        "face_width_ratio_max": 1.25,
        "face_height_ratio_min": 0.75,
        "face_height_ratio_max": 1.25,
        "face_aspect_relative_delta_max": 0.25,
        "center_shift_source_width_max": 0.18,
        "center_shift_source_height_max": 0.18,
        "eye_axis_roll_delta_degrees_max": 12.0,
    }
    failures: list[str] = []
    source_count, result_count = len(source_faces), len(result_faces)
    if source_count == 0:
        failures.append("source face detector found no face")
    if result_count == 0:
        failures.append("result face detector found no face")
    if result_count != source_count:
        failures.append(
            f"detected face count changed from {source_count} to {result_count}; possible duplicate or lost face"
        )
    if source_index >= source_count:
        failures.append("selected source face index is out of range")
    if result_index >= result_count:
        failures.append("selected result face index is out of range")

    source_geometry = None
    result_geometry = None
    comparisons = None
    if source_index < source_count and result_index < result_count:
        source_geometry = _face_geometry(source_faces[source_index])
        result_geometry = _face_geometry(result_faces[result_index])
        width_ratio = result_geometry["width"] / source_geometry["width"]
        height_ratio = result_geometry["height"] / source_geometry["height"]
        aspect_delta = abs(
            result_geometry["aspect_ratio"] - source_geometry["aspect_ratio"]
        ) / max(source_geometry["aspect_ratio"], 1.0e-6)
        center_shift_x = abs(result_geometry["center_x"] - source_geometry["center_x"]) / source_geometry["width"]
        center_shift_y = abs(result_geometry["center_y"] - source_geometry["center_y"]) / source_geometry["height"]
        roll_delta = _angle_distance_degrees(
            source_geometry["eye_axis_roll_degrees"], result_geometry["eye_axis_roll_degrees"]
        )
        comparisons = {
            "face_width_ratio": width_ratio,
            "face_height_ratio": height_ratio,
            "face_area_ratio": result_geometry["area"] / source_geometry["area"],
            "face_aspect_relative_delta": aspect_delta,
            "center_shift_source_width": center_shift_x,
            "center_shift_source_height": center_shift_y,
            "eye_axis_roll_delta_degrees": roll_delta,
        }
        if not thresholds["face_width_ratio_min"] <= width_ratio <= thresholds["face_width_ratio_max"]:
            failures.append(f"face width ratio {width_ratio:.3f} is outside the preserved-photo range")
        if not thresholds["face_height_ratio_min"] <= height_ratio <= thresholds["face_height_ratio_max"]:
            failures.append(f"face height ratio {height_ratio:.3f} is outside the preserved-photo range")
        if aspect_delta > thresholds["face_aspect_relative_delta_max"]:
            failures.append(f"face aspect-ratio delta {aspect_delta:.3f} is too large")
        if center_shift_x > thresholds["center_shift_source_width_max"]:
            failures.append(f"horizontal face-center shift {center_shift_x:.3f} is too large")
        if center_shift_y > thresholds["center_shift_source_height_max"]:
            failures.append(f"vertical face-center shift {center_shift_y:.3f} is too large")
        if roll_delta > thresholds["eye_axis_roll_delta_degrees_max"]:
            failures.append(f"eye-axis roll changed by {roll_delta:.2f} degrees")

    return {
        "gate_name": "normal_photo_geometry_first",
        "gate_passed": not failures,
        "failures": failures,
        "source_detected_face_count": source_count,
        "result_detected_face_count": result_count,
        "source_selected_face_geometry": source_geometry,
        "result_selected_face_geometry": result_geometry,
        "comparisons": comparisons,
        "thresholds": thresholds,
        "automatic_scope": [
            "detected face-count preservation",
            "selected face scale and aspect preservation",
            "selected face center and roll preservation",
        ],
        "manual_visual_hard_gate_required": True,
        "manual_visual_checks_before_any_identity_review": [
            "normal head-to-body proportion",
            "continuous and plausible head-neck-shoulder anatomy",
            "coherent pose and perspective",
            "no duplicated head, face, reflection, portrait, or face-shaped background artifact",
            "looks like one normal photograph before judging identity or eye detail",
        ],
    }


class FaceLocalPromptQualityGate:
    """Reject obviously incomplete VLM captions before they reach Krea2.

    This is deliberately a format/coverage gate, not a claim that the caption
    is visually truthful.  The raw and gated text remain available for review.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "minimum_characters": ("INT", {"default": 120, "min": 40, "max": 2000}),
                "minimum_sentences": ("INT", {"default": 2, "min": 1, "max": 8}),
                "maximum_sentences": ("INT", {"default": 8, "min": 1, "max": 30}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("validated_prompt", "quality_report_json")
    FUNCTION = "validate"
    CATEGORY = "face local edit/3 prompt"

    def validate(self, prompt, minimum_characters=120, minimum_sentences=2, maximum_sentences=8):
        text = str(prompt).strip()
        sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
        lowered = text.lower()
        coverage = {
            "subject_or_face": any(term in lowered for term in ("face", "portrait", "woman", "person", "subject")),
            "head_pose_or_direction": any(term in lowered for term in ("head", "pose", "facing", "profile", "three-quarter")),
            "eyes_or_gaze": any(term in lowered for term in ("eye", "gaze", "looking")),
            "lighting": any(term in lowered for term in ("light", "lit", "illumination", "shadow")),
        }
        failures = []
        if len(text) < int(minimum_characters):
            failures.append(f"prompt has only {len(text)} characters")
        if sentence_count < int(minimum_sentences):
            failures.append(f"prompt has {sentence_count} sentences; expected at least {minimum_sentences}")
        missing = [name for name, present in coverage.items() if not present]
        if missing:
            failures.append("missing required visual coverage: " + ", ".join(missing))
        if failures:
            raise ValueError("Automatic local prompt quality gate failed: " + "; ".join(failures))
        report = {
            "gate_name": "automatic_local_prompt_format_and_coverage",
            "gate_passed": True,
            "character_count": len(text),
            "sentence_count": sentence_count,
            "preferred_maximum_sentences": int(maximum_sentences),
            "above_preferred_sentence_count": sentence_count > int(maximum_sentences),
            "coverage": coverage,
            "factual_grounding_still_requires_visual_review": True,
        }
        return text, json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalNonEmptyMaskGate:
    """Stop the run instead of silently returning the unchanged source image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "support_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "minimum_fraction": (
                    "FLOAT",
                    {"default": 0.005, "min": 0.0001, "max": 1.0, "step": 0.0001},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("validated_mask", "mask_report_json")
    FUNCTION = "validate"
    CATEGORY = "face local edit/2 mask"

    def validate(self, mask, support_threshold=0.001, minimum_fraction=0.005):
        if isinstance(mask, torch.Tensor):
            tensor = mask.detach().to(device="cpu", dtype=torch.float32)
        else:
            tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))
        if tensor.ndim == 4:
            tensor = tensor[..., 0] if tensor.shape[-1] == 1 else tensor.amax(dim=-1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"MASK must be BHW, received shape {tuple(tensor.shape)}")
        tensor = tensor.clamp(0.0, 1.0)
        fraction = float((tensor > float(support_threshold)).float().mean().item())
        if fraction < float(minimum_fraction):
            raise ValueError(
                "Semantic head mask is empty or too small: "
                f"support_fraction={fraction:.6f}, minimum={float(minimum_fraction):.6f}"
            )
        report = {
            "gate_name": "non_empty_semantic_head_mask",
            "gate_passed": True,
            "support_threshold": float(support_threshold),
            "support_fraction": fraction,
            "minimum_fraction": float(minimum_fraction),
        }
        return tensor, json.dumps(report, ensure_ascii=False, indent=2)


class FaceLocalStructureDelta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_landmarks": ("FACE_LANDMARKS",),
                "result_landmarks": ("FACE_LANDMARKS",),
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 15}),
                "result_face_index": ("INT", {"default": 0, "min": 0, "max": 15}),
                "advisory_threshold": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.01, "max": 0.50, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("structure_delta_json", "groups_above_threshold", "automatic_support_only")
    FUNCTION = "compare"
    CATEGORY = "face local edit/6 audit"

    def compare(
        self,
        source_landmarks,
        result_landmarks,
        source_face_index,
        result_face_index,
        advisory_threshold,
    ):
        source_faces = _detected_faces(source_landmarks)
        result_faces = _detected_faces(result_landmarks)
        source_index, result_index = int(source_face_index), int(result_face_index)
        geometry_gate = _normal_photo_geometry_gate(
            source_faces, result_faces, source_index, result_index
        )
        if not geometry_gate["gate_passed"]:
            report = {
                "evaluation_order": [
                    "normal_photo_and_anatomy",
                    "source_state_and_scene_preservation",
                    "duplicate_face_boundary_and_composite",
                    "identity_transfer",
                    "eyes_lighting_and_fine_detail",
                ],
                "normal_photo_geometry_gate": geometry_gate,
                "normal_photo_geometry_gate_passed": False,
                "downstream_identity_evaluation_allowed": False,
                "advisory_threshold": float(advisory_threshold),
                "groups": {},
                "groups_above_threshold": 0,
                "minimum_structure_groups": 3,
                "automatic_support_only": False,
                "final_identity_verdict": "not_evaluated_hard_normal_photo_failure",
                "automatic_metrics_do_not_replace_human_acceptance": True,
            }
            return json.dumps(report, ensure_ascii=False, indent=2), 0, False
        source_metrics = _landmark_metrics(source_faces[source_index])
        result_metrics = _landmark_metrics(result_faces[result_index])
        threshold = float(advisory_threshold)
        groups: dict[str, Any] = {}
        passed = 0
        for group in STRUCTURE_GROUPS:
            metric_deltas = {}
            for name, source_value in source_metrics[group].items():
                result_value = result_metrics[group][name]
                denominator = max(abs(source_value), 1.0e-6)
                metric_deltas[name] = abs(result_value - source_value) / denominator
            maximum = max(metric_deltas.values())
            above = maximum >= threshold
            passed += int(above)
            groups[group] = {
                "source": source_metrics[group],
                "result": result_metrics[group],
                "relative_deltas": metric_deltas,
                "max_relative_delta": maximum,
                "above_advisory_threshold": above,
            }
        report = {
            "evaluation_order": [
                "normal_photo_and_anatomy",
                "source_state_and_scene_preservation",
                "duplicate_face_boundary_and_composite",
                "identity_transfer",
                "eyes_lighting_and_fine_detail",
            ],
            "normal_photo_geometry_gate": geometry_gate,
            "normal_photo_geometry_gate_passed": True,
            "downstream_identity_evaluation_allowed": True,
            "advisory_threshold": threshold,
            "groups": groups,
            "groups_above_threshold": passed,
            "minimum_structure_groups": 3,
            "automatic_support_only": passed >= 3,
            "final_identity_verdict": "requires_user_side_by_side_review",
            "automatic_metrics_do_not_replace_human_acceptance": True,
        }
        return json.dumps(report, ensure_ascii=False, indent=2), passed, passed >= 3


class SemanticObjectOptionalSAMPrompt:
    """Prepare an optional SAM prompt without translating an empty value.

    The protection branch is allowed to be blank.  Returning an explicit
    boolean from the same proxied text widget keeps branch selection and prompt
    translation in sync: a blank value prunes protection SAM before any Argos
    import, while a non-empty value follows the verified offline-only
    Chinese-to-English path.
    """

    CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "可留空；留空时不执行保护对象 SAM。中文使用已安装的 "
                            "Argos 模型离线翻译，英文原样通过。"
                        ),
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("sam_english_prompt", "has_text")
    FUNCTION = "translate"
    CATEGORY = "semantic object replacement/selection"
    DESCRIPTION = (
        "可为空的保护对象 SAM 语义输入；空白直接关闭保护分支，非空中文仅离线翻译。"
    )

    def translate(self, text):
        source = str(text).strip()
        if not source:
            return "", False
        if not self.CJK_PATTERN.search(source):
            return source, True

        try:
            import argostranslate.translate as argos_translate
        except ImportError as exc:
            raise ValueError(
                "未找到内建离线翻译组件 Argos Translate；SAM 中文提示词无法安全翻译"
            ) from exc

        languages = {
            language.code: language
            for language in argos_translate.get_installed_languages()
        }
        chinese = languages.get("zh")
        english = languages.get("en")
        if chinese is None or english is None:
            raise ValueError("未找到已安装的中文→英文离线翻译模型；不会自动联网下载")
        try:
            translation = chinese.get_translation(english)
            translated = str(translation.translate(source)).strip()
        except Exception as exc:
            raise ValueError(f"SAM 中文提示词离线翻译失败：{exc}") from exc
        if not translated or self.CJK_PATTERN.search(translated):
            raise ValueError(
                f"SAM 中文提示词未能完整翻译为英文：{translated or '<空>'}"
            )
        return translated, True


class SemanticObjectContextColorMatch:
    """Match generated low-frequency context to the source outside edit support.

    Klein image editing may reconstruct the whole crop with a different local
    exposure even when the prompt asks it to preserve context.  This node
    estimates only a smooth RGB correction field from untouched context pixels
    outside the editable support and applies that field to the generated crop.
    It does not copy source texture or alter the final replacement mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_local": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "editable_support_mask": ("MASK",),
                "correction_sigma": (
                    "INT",
                    {"default": 48, "min": 8, "max": 256, "step": 8},
                ),
                "maximum_correction": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "context_matched_generated",
        "smooth_correction_preview",
        "report_json",
    )
    FUNCTION = "match"
    CATEGORY = "semantic object replacement/color"

    def match(
        self,
        source_local,
        generated_local,
        editable_support_mask,
        correction_sigma=48,
        maximum_correction=0.35,
    ):
        source = _image_batch(source_local).to(device="cpu", dtype=torch.float32)
        generated = _image_batch(generated_local).to(
            device="cpu", dtype=torch.float32
        )
        if source.shape != generated.shape:
            raise ValueError(
                "source_local and generated_local must have identical dimensions"
            )
        height, width = int(source.shape[1]), int(source.shape[2])
        support = (
            _mask_batch(editable_support_mask, height, width)[0].numpy() > 0.001
        )
        known = ~support
        known_pixels = int(np.count_nonzero(known))
        if known_pixels < 256:
            raise ValueError(
                "editable_support_mask leaves fewer than 256 context pixels"
            )
        sigma = int(correction_sigma)
        limit = float(maximum_correction)
        source_np = source[0].numpy()
        generated_np = generated[0].numpy()
        weights = known.astype(np.float32)
        blurred_weight = ndimage.gaussian_filter(
            weights, sigma=sigma, mode="nearest"
        )
        if float(np.max(blurred_weight)) <= 1e-8:
            raise RuntimeError("context correction has no usable source weight")
        correction = np.zeros_like(source_np, dtype=np.float32)
        raw_delta = source_np - generated_np
        for channel in range(3):
            numerator = ndimage.gaussian_filter(
                raw_delta[..., channel] * weights,
                sigma=sigma,
                mode="nearest",
            )
            correction[..., channel] = numerator / np.maximum(
                blurred_weight, 1e-6
            )
        correction = np.clip(correction, -limit, limit)
        matched_np = np.clip(generated_np + correction, 0.0, 1.0)
        before_mae = float(np.mean(np.abs(raw_delta[known])))
        after_mae = float(np.mean(np.abs(source_np[known] - matched_np[known])))
        preview_np = np.clip(0.5 + correction, 0.0, 1.0)
        report = {
            "algorithm": "semantic-object-low-frequency-context-match-v1",
            "dimensions": [width, height],
            "known_context_pixels": known_pixels,
            "editable_support_pixels": int(np.count_nonzero(support)),
            "correction_sigma": sigma,
            "maximum_correction": limit,
            "known_context_mae_before": before_mae,
            "known_context_mae_after": after_mae,
            "source_texture_copied": False,
            "final_mask_modified": False,
        }
        return (
            torch.from_numpy(matched_np).unsqueeze(0),
            torch.from_numpy(preview_np).unsqueeze(0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class SemanticObjectDifferenceMaskExact:
    """Build a generic object-replacement mask from exact aligned crops.

    The semantic target is always retained.  Generated differences may expand
    beyond the old object outline, but only inside the explicit editable
    support.  Protection is subtracted last and therefore always wins.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_local": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "target_core_mask": ("MASK",),
                "protection_mask": ("MASK",),
                "editable_support_mask": ("MASK",),
                "threshold_level": (
                    "INT",
                    {"default": 7, "min": 1, "max": 255, "step": 1},
                ),
                "difference_expand": (
                    "INT",
                    {"default": 4, "min": 0, "max": 256, "step": 1},
                ),
            },
            "optional": {
                "contract_version": (
                    "STRING",
                    {"default": "semantic-object-protection-overlap-v2"},
                )
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "automatic_replacement_mask",
        "bounded_difference_mask",
        "red_cyan_difference_preview",
        "report_json",
    )
    FUNCTION = "build"
    CATEGORY = "semantic object replacement/mask"

    def build(
        self,
        source_local,
        generated_local,
        target_core_mask,
        protection_mask,
        editable_support_mask,
        threshold_level=7,
        difference_expand=4,
        contract_version="semantic-object-protection-overlap-v2",
    ):
        if str(contract_version) != "semantic-object-protection-overlap-v2":
            raise ValueError("unsupported semantic object mask contract version")
        source = _image_batch(source_local).to(device="cpu", dtype=torch.float32)
        generated = _image_batch(generated_local).to(
            device="cpu", dtype=torch.float32
        )
        if source.shape != generated.shape:
            raise ValueError(
                "source_local and generated_local must have identical dimensions"
            )
        height, width = int(source.shape[1]), int(source.shape[2])
        target = _mask_batch(target_core_mask, height, width)[0].numpy() > 0.001
        protection = (
            _mask_batch(protection_mask, height, width)[0].numpy() > 0.001
        )
        support = (
            _mask_batch(editable_support_mask, height, width)[0].numpy() > 0.001
        )
        if not np.any(target):
            raise ValueError("target_core_mask is empty")
        effective_target = target & ~protection
        if not np.any(effective_target):
            raise ValueError(
                "protection_mask removes the complete target core"
            )
        if np.any(effective_target & ~support):
            raise ValueError(
                "unprotected target core extends outside editable_support_mask"
            )

        threshold = int(threshold_level)
        source_u8 = torch.round(source.clamp(0.0, 1.0) * 255.0).to(torch.int16)
        generated_u8 = torch.round(
            generated.clamp(0.0, 1.0) * 255.0
        ).to(torch.int16)
        raw_difference = (
            torch.amax(torch.abs(generated_u8 - source_u8), dim=-1)[0].numpy()
            >= threshold
        )
        expand = int(difference_expand)
        if expand:
            raw_difference = ndimage.binary_dilation(
                raw_difference,
                structure=_disk_structure(expand),
            )
        bounded_difference = raw_difference & support & ~protection
        automatic = (effective_target | bounded_difference) & support & ~protection
        if not np.any(automatic):
            raise ValueError(
                "protection_mask removes the complete replacement mask"
            )

        luminance_weights = torch.tensor(
            (0.299, 0.587, 0.114), dtype=torch.float32
        ).view(1, 1, 1, 3)
        source_gray = torch.sum(source * luminance_weights, dim=-1, keepdim=True)
        generated_gray = torch.sum(
            generated * luminance_weights, dim=-1, keepdim=True
        )
        red_cyan = torch.cat(
            (source_gray, generated_gray, generated_gray), dim=-1
        ).clamp(0.0, 1.0)
        report = {
            "algorithm": "semantic-object-core-plus-bounded-rgb-difference-v1",
            "contract_version": "semantic-object-protection-overlap-v2",
            "dimensions": [width, height],
            "difference_metric": "max(abs(ai_u8_rgb - source_u8_rgb))",
            "threshold_level": threshold,
            "difference_expand": expand,
            "pixels": {
                "target_core": int(np.count_nonzero(target)),
                "effective_target_core": int(np.count_nonzero(effective_target)),
                "editable_support": int(np.count_nonzero(support)),
                "protection": int(np.count_nonzero(protection)),
                "bounded_difference": int(np.count_nonzero(bounded_difference)),
                "automatic_replacement": int(np.count_nonzero(automatic)),
                "automatic_outside_support": 0,
                "automatic_inside_protection": 0,
            },
            "target_core_preserved_except_protection": True,
            "difference_clipped_to_editable_support": True,
            "protection_wins": True,
        }
        return (
            torch.from_numpy(automatic.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(bounded_difference.astype(np.float32)).unsqueeze(0),
            red_cyan,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class SemanticObjectEditorSizePlan:
    """Clamp editor input to a total-pixel range while preserving aspect."""

    MAX_EDITOR_LONG_SIDE = 2048

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.10, "max": 8.0, "step": 0.05},
                ),
                "max_megapixels": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.10, "max": 8.0, "step": 0.05},
                ),
                "multiple": (
                    "INT",
                    {"default": 16, "min": 8, "max": 64, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "original_width",
        "original_height",
        "editor_width",
        "editor_height",
        "resize_required",
        "report_json",
    )
    FUNCTION = "plan"
    CATEGORY = "semantic object replacement/crop"

    def plan(
        self,
        image,
        min_megapixels=1.0,
        max_megapixels=1.5,
        multiple=16,
    ):
        tensor = _image_batch(image)
        original_height, original_width = int(tensor.shape[1]), int(tensor.shape[2])
        alignment = int(multiple)
        if alignment <= 0:
            raise ValueError("multiple must be positive")
        minimum_mp = float(min_megapixels)
        maximum_mp = float(max_megapixels)
        if not math.isfinite(minimum_mp) or not math.isfinite(maximum_mp):
            raise ValueError("megapixel limits must be finite")
        if minimum_mp <= 0.0 or maximum_mp <= 0.0:
            raise ValueError("megapixel limits must be positive")
        if minimum_mp > maximum_mp:
            raise ValueError(
                "min_megapixels must not exceed max_megapixels"
            )

        megapixel_unit = 1024 * 1024
        minimum_pixels = int(round(minimum_mp * megapixel_unit))
        maximum_pixels = int(round(maximum_mp * megapixel_unit))
        original_pixels = original_width * original_height
        longer_side = max(original_width, original_height)
        shorter_side = min(original_width, original_height)
        maximum_long_side = self.MAX_EDITOR_LONG_SIDE
        maximum_pixels_at_safe_long_side = int(
            round(
                maximum_long_side
                * maximum_long_side
                * shorter_side
                / float(longer_side)
            )
        )
        if maximum_pixels_at_safe_long_side < minimum_pixels:
            raise ValueError(
                "crop aspect ratio is too extreme for the configured "
                f"{minimum_mp:.2f}MP floor and {maximum_long_side}px "
                "single-axis safety limit; increase crop context"
            )

        if original_pixels < minimum_pixels:
            target_pixels = minimum_pixels
            requested_direction = "upscale"
        elif original_pixels > maximum_pixels:
            target_pixels = maximum_pixels
            requested_direction = "downscale"
        else:
            target_pixels = original_pixels
            requested_direction = "none"

        ideal_scale = math.sqrt(target_pixels / float(original_pixels))
        if longer_side * ideal_scale > maximum_long_side:
            ideal_scale = maximum_long_side / float(longer_side)
            target_pixels = int(round(original_pixels * ideal_scale * ideal_scale))
            requested_direction = "downscale"

        if (
            requested_direction == "none"
            and original_width % alignment == 0
            and original_height % alignment == 0
            and longer_side <= maximum_long_side
        ):
            editor_width = original_width
            editor_height = original_height
        else:
            ideal_width = original_width * ideal_scale
            ideal_height = original_height * ideal_scale
            if min(ideal_width, ideal_height) < alignment / 2.0:
                raise ValueError(
                    "crop aspect ratio is too extreme for multiple="
                    f"{alignment}; increase crop context"
                )
            # Match ComfyUI's mature ImageScaleToTotalPixels rule used by
            # the face workflow: independently round both dimensions to
            # the requested resolution step. This keeps aspect error lower
            # than forcing one axis outward just to cross an exact pixel
            # boundary after discrete alignment.
            editor_width = max(
                alignment,
                int(round(ideal_width / alignment)) * alignment,
            )
            editor_height = max(
                alignment,
                int(round(ideal_height / alignment)) * alignment,
            )
        if max(editor_width, editor_height) > maximum_long_side:
            raise RuntimeError("planned editor size exceeds single-axis limit")

        editor_pixels = editor_width * editor_height
        alignment_tolerance_pixels = alignment * max(
            editor_width,
            editor_height,
        )
        lower_with_tolerance = minimum_pixels - alignment_tolerance_pixels
        upper_with_tolerance = maximum_pixels + alignment_tolerance_pixels
        if not lower_with_tolerance <= editor_pixels <= upper_with_tolerance:
            raise RuntimeError(
                "planned editor size exceeds alignment-tolerant pixel bounds"
            )
        resize_required = (editor_width, editor_height) != (
            original_width,
            original_height,
        )
        if not resize_required:
            resize_direction = "none"
        elif editor_pixels > original_pixels:
            resize_direction = "upscale"
        elif editor_pixels < original_pixels:
            resize_direction = "downscale"
        else:
            resize_direction = "alignment_only"
        report = {
            "algorithm": "semantic-object-bounded-total-pixel-size-plan-v2",
            "original_width": original_width,
            "original_height": original_height,
            "original_pixels": original_pixels,
            "original_megapixels": original_pixels / float(megapixel_unit),
            "editor_width": editor_width,
            "editor_height": editor_height,
            "editor_pixels": editor_pixels,
            "editor_megapixels": editor_pixels / float(megapixel_unit),
            "min_megapixels": minimum_mp,
            "max_megapixels": maximum_mp,
            "minimum_pixels": minimum_pixels,
            "maximum_pixels": maximum_pixels,
            "target_pixels_before_alignment": target_pixels,
            "multiple": alignment,
            "maximum_long_side": maximum_long_side,
            "alignment_tolerance_pixels": alignment_tolerance_pixels,
            "resize_required": resize_required,
            "resize_direction": resize_direction,
            "small_crop_upscaled": resize_direction == "upscale",
            "large_crop_downscaled": resize_direction == "downscale",
            "strict_pixel_range_satisfied": (
                minimum_pixels <= editor_pixels <= maximum_pixels
            ),
            "pixel_range_satisfied_with_alignment_tolerance": True,
            "aspect_ratio_relative_error": abs(
                (editor_width / float(editor_height))
                / (original_width / float(original_height))
                - 1.0
            ),
        }
        return (
            original_width,
            original_height,
            editor_width,
            editor_height,
            resize_required,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class SemanticObjectMaskEditorCanvas(FaceLocalMaskEditorCanvas):
    CATEGORY = "semantic object replacement/mask"


class SemanticObjectManualMaskCorrection:
    """Apply a complete generic-object MaskEditor correction.

    Impact Pack's PreviewBridge emits a 64x64 placeholder before MaskEditor is
    saved.  Only that sentinel falls back to ``automatic_mask``.  A same-size
    edited mask, including an empty one, is an explicit complete replacement.
    There is no mandatory-core lock, so a user can erase a SAM false-positive.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "difference_image": ("IMAGE",),
                "automatic_mask": ("MASK",),
                "edited_mask": ("MASK",),
                "processing_support_mask": ("MASK",),
                "support_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "final_mask",
        "manual_add_mask",
        "manual_erase_mask",
        "correction_preview",
        "report_json",
    )
    FUNCTION = "correct"
    CATEGORY = "semantic object replacement/mask"

    @staticmethod
    def _raw_mask_size(mask_value) -> tuple[int, int]:
        value = (
            mask_value.detach().to(device="cpu", dtype=torch.float32)
            if isinstance(mask_value, torch.Tensor)
            else torch.from_numpy(np.asarray(mask_value, dtype=np.float32))
        )
        if value.ndim == 2:
            return int(value.shape[0]), int(value.shape[1])
        if value.ndim == 3:
            return int(value.shape[-2]), int(value.shape[-1])
        if value.ndim == 4:
            if value.shape[-1] == 1:
                return int(value.shape[1]), int(value.shape[2])
            if value.shape[1] == 1:
                return int(value.shape[2]), int(value.shape[3])
            return int(value.shape[1]), int(value.shape[2])
        raise ValueError(f"MASK must be BHW, received shape {tuple(value.shape)}")

    def correct(
        self,
        difference_image,
        automatic_mask,
        edited_mask,
        processing_support_mask,
        support_threshold=0.001,
    ):
        difference = _image_batch(difference_image).to(
            device="cpu", dtype=torch.float32
        )
        if int(difference.shape[0]) != 1:
            raise ValueError("generic manual mask correction requires image batch size 1")
        height, width = int(difference.shape[1]), int(difference.shape[2])
        expected_size = (height, width)
        raw_editor_size = self._raw_mask_size(edited_mask)
        placeholder_fallback = bool(
            raw_editor_size == (64, 64) and raw_editor_size != expected_size
        )
        if raw_editor_size != expected_size and not placeholder_fallback:
            raise ValueError(
                "edited_mask dimensions do not match the correction canvas: "
                f"received {raw_editor_size[1]}x{raw_editor_size[0]}, "
                f"expected {width}x{height}"
            )

        threshold = float(support_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        automatic = _mask_batch(automatic_mask, height, width)
        support = _mask_batch(processing_support_mask, height, width)
        support_selected = support > threshold
        if not torch.any(support_selected):
            raise ValueError("processing_support_mask is empty")

        if placeholder_fallback:
            edited = automatic.clone()
            manual_editor_applied = False
        else:
            edited = _mask_batch(edited_mask, height, width)
            manual_editor_applied = True

        automatic_in_support = torch.where(
            support_selected, automatic, torch.zeros_like(automatic)
        )
        edited_in_support = torch.where(
            support_selected, edited, torch.zeros_like(edited)
        )
        final_mask = edited_in_support.clone()
        final_mask[final_mask < threshold] = 0.0
        raw_add = torch.clamp(
            edited_in_support - automatic_in_support, 0.0, 1.0
        )
        raw_erase = torch.clamp(
            automatic_in_support - edited_in_support, 0.0, 1.0
        )
        manual_add = raw_add.masked_fill(raw_add < threshold, 0.0)
        manual_erase = raw_erase.masked_fill(raw_erase < threshold, 0.0)

        preview = difference.clone()
        layers = (
            (final_mask, (1.0, 0.12, 0.08), 0.34),
            (manual_add, (0.05, 1.0, 0.15), 0.72),
            (manual_erase, (1.0, 0.05, 0.85), 0.72),
        )
        for layer, color_values, opacity in layers:
            weight = torch.clamp(
                layer.unsqueeze(-1) * float(opacity), 0.0, float(opacity)
            )
            color = torch.tensor(color_values, dtype=torch.float32).view(
                1, 1, 1, 3
            )
            preview = preview * (1.0 - weight) + color * weight

        outside_final_zero = bool(
            torch.all(final_mask[~support_selected] == 0.0)
        )
        outside_add_zero = bool(
            torch.all(manual_add[~support_selected] == 0.0)
        )
        outside_erase_zero = bool(
            torch.all(manual_erase[~support_selected] == 0.0)
        )
        expected_final = (
            automatic_in_support if placeholder_fallback else edited_in_support
        ).clone()
        expected_final[expected_final < threshold] = 0.0
        fallback_or_override_exact = bool(torch.equal(final_mask, expected_final))
        add_formula_exact = bool(torch.equal(manual_add, raw_add.masked_fill(raw_add < threshold, 0.0)))
        erase_formula_exact = bool(
            torch.equal(
                manual_erase, raw_erase.masked_fill(raw_erase < threshold, 0.0)
            )
        )
        gate_passed = bool(
            outside_final_zero
            and outside_add_zero
            and outside_erase_zero
            and fallback_or_override_exact
            and add_formula_exact
            and erase_formula_exact
        )
        report = {
            "algorithm": "semantic-object-complete-manual-mask-correction-v1",
            "editor_canvas_initialized": not placeholder_fallback,
            "previewbridge_placeholder_fallback": placeholder_fallback,
            "manual_editor_applied": manual_editor_applied,
            "raw_editor_size_hw": list(raw_editor_size),
            "expected_editor_size_hw": [height, width],
            "complete_edited_mask_replaces_automatic": manual_editor_applied,
            "automatic_support_pixels": int(
                torch.count_nonzero(automatic_in_support > threshold).item()
            ),
            "edited_support_pixels": int(
                torch.count_nonzero(edited_in_support > threshold).item()
            ),
            "manual_add_pixels": int(
                torch.count_nonzero(manual_add > threshold).item()
            ),
            "manual_erase_pixels": int(
                torch.count_nonzero(manual_erase > threshold).item()
            ),
            "final_support_pixels": int(
                torch.count_nonzero(final_mask > threshold).item()
            ),
            "outside_processing_support": {
                "final_is_zero": outside_final_zero,
                "manual_add_is_zero": outside_add_zero,
                "manual_erase_is_zero": outside_erase_zero,
            },
            "manual_core_erasure_allowed": True,
            "fallback_or_override_exact": fallback_or_override_exact,
            "add_formula_exact": add_formula_exact,
            "erase_formula_exact": erase_formula_exact,
            "preview_legend": {
                "red": "final replacement mask",
                "green": "manual add",
                "magenta": "manual erase",
            },
            "gate_passed": gate_passed,
        }
        return (
            final_mask,
            manual_add,
            manual_erase,
            preview.clamp(0.0, 1.0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class SemanticObjectReplacementPromptContract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "replacement_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "把选中的物品替换成目标物，保持其他内容不变。",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "klein_prompt",
        "gpt_text_only_prompt",
        "gpt_reference_prompt",
    )
    FUNCTION = "build"
    CATEGORY = "semantic object replacement/prompt"

    def build(self, replacement_instruction):
        instruction = str(replacement_instruction).strip()
        if not instruction:
            raise ValueError("replacement_instruction must not be empty")
        reference_scope = (
            " If Image 2 is provided, use it only as a visual reference for "
            "the replacement object. Do not copy Image 2's background, "
            "composition, people, lighting, shadows, or surface."
        )
        preservation = (
            " Replace only the selected target object. Preserve every other "
            "object, person, hand, surface, background element, composition, "
            "lighting, perspective, and texture in Image 1."
        )
        klein = instruction + reference_scope + preservation
        gpt_text_only = (
            "Image 1 is the base crop to edit. " + instruction + preservation
        )
        gpt_reference = (
            "Image 1 is the base crop to edit."
            + reference_scope
            + " "
            + instruction
            + preservation
        )
        return klein, gpt_text_only, gpt_reference


class SemanticObjectStrictCompositeExact:
    """Precompose a broad generated support, then restore one exact crop.

    ``selected_local_mask`` is a full-strength core, not the paste boundary.
    The complete writeback support is generated, and the surrounding
    generation-minus-writeback band smoothly transitions to the exact source
    crop.  Protection and explicit manual erasure are applied last.  Finally,
    an independent four-side crop-perimeter ramp restores the clean local crop
    to its exact original coordinates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "source_local": ("IMAGE",),
                "generated_local": ("IMAGE",),
                "selected_local_mask": ("MASK",),
                "target_core_mask": ("MASK",),
                "generation_support_mask": ("MASK",),
                "writeback_support_mask": ("MASK",),
                "protection_mask": ("MASK",),
                "manual_erase_mask": ("MASK",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "top_feather_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "bottom_feather_percent": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "left_feather_percent": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "right_feather_percent": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "final_image",
        "strict_full_resolution_mask",
        "full_resolution_alpha",
        "clean_local_crop",
        "wide_support_alpha",
        "outer_crop_ramp",
        "difference_preview",
        "report_json",
    )
    FUNCTION = "composite"
    CATEGORY = "semantic object replacement/composite"

    def composite(
        self,
        original,
        source_local,
        generated_local,
        selected_local_mask,
        target_core_mask,
        generation_support_mask,
        writeback_support_mask,
        protection_mask,
        manual_erase_mask,
        x,
        y,
        width,
        height,
        top_feather_percent=10.0,
        bottom_feather_percent=10.0,
        left_feather_percent=5.0,
        right_feather_percent=5.0,
    ):
        base = _image_batch(original).to(device="cpu", dtype=torch.float32)
        source = _image_batch(source_local).to(device="cpu", dtype=torch.float32)
        candidate = _image_batch(generated_local).to(
            device="cpu", dtype=torch.float32
        )
        if int(base.shape[0]) != 1 or int(source.shape[0]) != 1 or int(candidate.shape[0]) != 1:
            raise ValueError("strict single-crop composite requires image batch size 1")

        full_height, full_width = int(base.shape[1]), int(base.shape[2])
        px, py, crop_width, crop_height = int(x), int(y), int(width), int(height)
        exact_local_shape = (crop_height, crop_width)
        if tuple(source.shape[1:3]) != exact_local_shape:
            raise ValueError(
                "source_local dimensions do not match the exact crop size"
            )
        if tuple(candidate.shape[1:3]) != exact_local_shape:
            raise ValueError(
                "generated_local dimensions do not match the exact crop size"
            )
        if px < 0 or py < 0 or px + crop_width > full_width or py + crop_height > full_height:
            raise ValueError(
                "exact crop coordinates extend outside the original image"
            )
        exact_source_from_original = base[
            :, py : py + crop_height, px : px + crop_width, :
        ]
        source_matches_original = torch.equal(
            source,
            exact_source_from_original,
        )
        if not source_matches_original:
            raise ValueError(
                "source_local must exactly equal the original image at the "
                "declared crop coordinates"
            )

        quantization_floor = 0.5 / 255.0
        selected = (
            _mask_batch(selected_local_mask, crop_height, crop_width)[0]
            .numpy()
            > quantization_floor
        )
        core = (
            _mask_batch(target_core_mask, crop_height, crop_width)[0].numpy()
            > quantization_floor
        )
        generation = (
            _mask_batch(generation_support_mask, crop_height, crop_width)[0].numpy()
            > quantization_floor
        )
        writeback = (
            _mask_batch(writeback_support_mask, crop_height, crop_width)[0].numpy()
            > quantization_floor
        )
        protection = (
            _mask_batch(protection_mask, crop_height, crop_width)[0].numpy()
            > quantization_floor
        )
        manual_erase = (
            _mask_batch(manual_erase_mask, crop_height, crop_width)[0].numpy()
            > quantization_floor
        )
        if not np.any(generation):
            raise ValueError("generation_support_mask is empty")
        if not np.any(writeback):
            raise ValueError("writeback_support_mask is empty")
        if np.any(writeback & ~generation):
            raise ValueError(
                "writeback support must be fully contained by generation support"
            )
        if np.any(core & ~generation):
            raise ValueError(
                "target core must be fully contained by generation support"
            )
        if np.any(selected & ~generation):
            raise ValueError(
                "selected replacement core must be fully contained by "
                "generation support"
            )
        keep_original = protection | manual_erase
        effective_selected = selected & ~keep_original
        if not np.any(effective_selected):
            raise ValueError(
                "protection/manual erase removes the complete selected replacement core"
            )

        # The whole writeback support is generated.  Its surrounding ring fades
        # according to the relative distance from writeback and to the outside
        # of generation support.  The semantic selection is never used as this
        # transition boundary.
        wide_alpha = np.zeros((crop_height, crop_width), dtype=np.float32)
        wide_alpha[writeback] = 1.0
        transition_ring = generation & ~writeback
        transition_values = np.zeros_like(wide_alpha)
        if np.any(transition_ring):
            distance_from_writeback = ndimage.distance_transform_edt(
                ~writeback
            ).astype(np.float32)
            padded_generation = np.pad(
                generation,
                ((1, 1), (1, 1)),
                mode="constant",
                constant_values=False,
            )
            distance_to_generation_outside = ndimage.distance_transform_edt(
                padded_generation
            )[1:-1, 1:-1].astype(np.float32)
            denominator = (
                distance_from_writeback + distance_to_generation_outside
            )
            ratio = np.zeros_like(wide_alpha)
            valid = transition_ring & (denominator > 0.0)
            ratio[valid] = (
                distance_to_generation_outside[valid] / denominator[valid]
            )
            transition_values[transition_ring] = _smoothstep_numpy(
                0.0, 1.0, ratio[transition_ring]
            )
            wide_alpha[transition_ring] = transition_values[transition_ring]
        wide_alpha[~generation] = 0.0
        wide_alpha[effective_selected] = 1.0
        writeback_full_before_keep = bool(np.all(wide_alpha[writeback] == 1.0))
        wide_alpha[keep_original] = 0.0

        wide_tensor = torch.from_numpy(wide_alpha.copy()).unsqueeze(0)
        wide_support = wide_tensor > quantization_floor
        wide_image = wide_tensor.unsqueeze(-1)
        clean_local = source.clone()
        local_blend = source * (1.0 - wide_image) + candidate * wide_image
        clean_local[wide_support] = local_blend[wide_support]
        clean_outside_exact = torch.equal(
            clean_local[~wide_support], source[~wide_support]
        )
        if not clean_outside_exact:
            raise RuntimeError(
                "local precompose changed pixels outside the wide support alpha"
            )

        side_percent = {
            "top": float(top_feather_percent),
            "bottom": float(bottom_feather_percent),
            "left": float(left_feather_percent),
            "right": float(right_feather_percent),
        }
        if any(value < 0.0 or value > 30.0 for value in side_percent.values()):
            raise ValueError("crop-perimeter feather percentages must be within 0–30")
        touches_original_boundary = {
            "top": py == 0,
            "bottom": py + crop_height == full_height,
            "left": px == 0,
            "right": px + crop_width == full_width,
        }
        requested_side_pixels = {
            "top": int(round(crop_height * side_percent["top"] / 100.0)),
            "bottom": int(round(crop_height * side_percent["bottom"] / 100.0)),
            "left": int(round(crop_width * side_percent["left"] / 100.0)),
            "right": int(round(crop_width * side_percent["right"] / 100.0)),
        }
        effective_side_pixels = {
            side: (
                0 if touches_original_boundary[side] else requested_side_pixels[side]
            )
            for side in ("top", "bottom", "left", "right")
        }

        yy, xx = np.indices((crop_height, crop_width), dtype=np.float32)
        side_distance = {
            "top": yy,
            "bottom": float(crop_height - 1) - yy,
            "left": xx,
            "right": float(crop_width - 1) - xx,
        }
        outer_ramp = np.ones((crop_height, crop_width), dtype=np.float32)
        for side, pixels in effective_side_pixels.items():
            if pixels <= 0:
                continue
            side_ramp = _smoothstep_numpy(
                0.0,
                float(pixels),
                side_distance[side],
            )
            outer_ramp = np.minimum(outer_ramp, side_ramp)
        outer_tensor = torch.from_numpy(outer_ramp.copy()).unsqueeze(0)
        # The physical crop perimeter is the final spatial constraint.
        # Semantic/difference/manual additions may expand inside the crop, but
        # must never punch a hard, full-alpha hole through this four-side ramp.
        final_local_alpha = wide_tensor * outer_tensor
        final_local_alpha[
            torch.from_numpy(keep_original).unsqueeze(0)
        ] = 0.0
        selected_index = torch.from_numpy(effective_selected).unsqueeze(0)
        selected_respects_crop_ramp = bool(
            torch.equal(
                final_local_alpha[selected_index],
                outer_tensor[selected_index],
            )
        )
        final_local_alpha = torch.where(
            final_local_alpha > quantization_floor,
            final_local_alpha,
            torch.zeros_like(final_local_alpha),
        )
        final_local_support = final_local_alpha > 0.0
        if int(torch.count_nonzero(final_local_support).item()) == 0:
            raise ValueError("the final two-stage composite alpha is empty")

        result = base.clone()
        destination = result[
            :, py : py + crop_height, px : px + crop_width, :
        ]
        outer_image = outer_tensor.unsqueeze(-1)
        outer_blend = destination * (1.0 - outer_image) + clean_local * outer_image
        destination[final_local_support] = outer_blend[final_local_support]

        full_alpha = torch.zeros(
            (1, full_height, full_width), dtype=torch.float32
        )
        full_alpha[
            :, py : py + crop_height, px : px + crop_width
        ] = final_local_alpha
        full_support = full_alpha > 0.0
        outside_equal = torch.equal(result[~full_support], base[~full_support])
        if not outside_equal:
            raise RuntimeError(
                "strict composite changed pixels outside the replacement mask"
            )
        local_full_support = full_support[
            :, py : py + crop_height, px : px + crop_width
        ]
        protection_overlap_pixels = int(
            torch.count_nonzero(
                local_full_support & torch.from_numpy(protection).unsqueeze(0)
            ).item()
        )
        manual_erase_overlap_pixels = int(
            torch.count_nonzero(
                local_full_support & torch.from_numpy(manual_erase).unsqueeze(0)
            ).item()
        )
        if protection_overlap_pixels:
            raise RuntimeError("strict composite support overlaps protection")
        if manual_erase_overlap_pixels:
            raise RuntimeError("strict composite support overlaps manual erase")
        result_local = result[
            :, py : py + crop_height, px : px + crop_width, :
        ]
        protection_tensor = torch.from_numpy(protection).unsqueeze(0)
        manual_erase_tensor = torch.from_numpy(manual_erase).unsqueeze(0)
        protection_is_exact_source = bool(
            torch.equal(result_local[protection_tensor], source[protection_tensor])
        )
        manual_erase_is_exact_source = bool(
            torch.equal(
                result_local[manual_erase_tensor], source[manual_erase_tensor]
            )
        )
        dimensions_equal = result.shape == base.shape
        if not dimensions_equal:
            raise RuntimeError("strict composite changed the original dimensions")

        difference = torch.abs(result - base)
        outside_pixel_difference = torch.any(
            difference > 0.0, dim=-1
        ) & ~full_support
        outside_mismatch_pixels = int(
            torch.count_nonzero(outside_pixel_difference).item()
        )
        outside_max_abs_diff = (
            float(difference[~full_support].max().item())
            if bool(torch.any(~full_support).item())
            else 0.0
        )
        generation_contains_writeback = not bool(np.any(writeback & ~generation))
        selected_inside_generation = not bool(np.any(selected & ~generation))
        target_core_inside_generation = not bool(np.any(core & ~generation))
        ring_values = wide_alpha[transition_ring & ~keep_original]
        ring_gradient_present = bool(
            np.any(
                (ring_values > quantization_floor)
                & (ring_values < 1.0 - quantization_floor)
            )
        )
        wide_support_pixels = int(torch.count_nonzero(wide_support).item())
        selected_pixels = int(np.count_nonzero(effective_selected))
        support_is_not_selected_outline = bool(
            wide_support_pixels > selected_pixels
        )
        gate_passed = bool(
            dimensions_equal
            and source_matches_original
            and clean_outside_exact
            and outside_equal
            and outside_mismatch_pixels == 0
            and generation_contains_writeback
            and selected_inside_generation
            and target_core_inside_generation
            and writeback_full_before_keep
            and protection_overlap_pixels == 0
            and manual_erase_overlap_pixels == 0
            and protection_is_exact_source
            and manual_erase_is_exact_source
            and selected_respects_crop_ramp
            and support_is_not_selected_outline
        )
        report = {
            "algorithm": "semantic-object-wide-support-strict-composite-v4",
            "support_alpha_formula": (
                "writeback=1; generation-minus-writeback=distance-ratio "
                "smoothstep; outside-generation=0"
            ),
            "generation_contains_writeback": generation_contains_writeback,
            "selected_inside_generation": selected_inside_generation,
            "target_core_inside_generation": target_core_inside_generation,
            "writeback_is_full_alpha_before_keep_masks": writeback_full_before_keep,
            "transition_ring_has_fractional_alpha": ring_gradient_present,
            "wide_support_is_not_selected_outline": support_is_not_selected_outline,
            "selected_pixels_respect_crop_perimeter_ramp": (
                selected_respects_crop_ramp
            ),
            "protection_pixels_are_exact_source": protection_is_exact_source,
            "manual_erase_pixels_are_exact_source": manual_erase_is_exact_source,
            "source_local_is_exact_original_crop": source_matches_original,
            "clean_local_outside_wide_support_is_exact_source": clean_outside_exact,
            "input_width": full_width,
            "input_height": full_height,
            "output_width": int(result.shape[2]),
            "output_height": int(result.shape[1]),
            "dimensions_equal": dimensions_equal,
            "crop_xywh": [px, py, crop_width, crop_height],
            "per_side_percent": side_percent,
            "requested_per_side_pixels": requested_side_pixels,
            "effective_per_side_pixels": effective_side_pixels,
            "touches_original_image_boundary": touches_original_boundary,
            "pixels": {
                "target_core": int(np.count_nonzero(core)),
                "generation_support": int(np.count_nonzero(generation)),
                "writeback_support": int(np.count_nonzero(writeback)),
                "selected_core": int(np.count_nonzero(selected)),
                "effective_selected_core": selected_pixels,
                "wide_support": wide_support_pixels,
                "fractional_transition_ring": int(
                    np.count_nonzero(
                        (transition_values > quantization_floor)
                        & (transition_values < 1.0 - quantization_floor)
                    )
                ),
                "strict_full_resolution_support": int(
                    torch.count_nonzero(full_support).item()
                ),
                "protection": int(np.count_nonzero(protection)),
                "manual_erase": int(np.count_nonzero(manual_erase)),
                "strict_support_inside_protection": protection_overlap_pixels,
                "strict_support_inside_manual_erase": manual_erase_overlap_pixels,
            },
            "outside_mask_mismatch_pixels_float": outside_mismatch_pixels,
            "outside_mask_max_abs_diff_float": outside_max_abs_diff,
            "outside_mask_is_exact_original": outside_equal,
            "internal_tile_count": 1,
            "alpha_quantization_floor": quantization_floor,
            "gate_passed": gate_passed,
        }
        return (
            result.clamp(0.0, 1.0),
            full_support.float(),
            full_alpha,
            clean_local.clamp(0.0, 1.0),
            wide_tensor,
            outer_tensor,
            difference.clamp(0.0, 1.0),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
