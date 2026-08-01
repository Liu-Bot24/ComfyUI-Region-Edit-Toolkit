from __future__ import annotations

import json

import numpy as np
import torch

from .. import nodes as smart_nodes
from . import face_object_nodes as face_nodes


ROOT_CATEGORY = "Region Edit Toolkit"


class RegionEditFaceSelect(face_nodes.FaceLocalSelectFace):
    CATEGORY = f"{ROOT_CATEGORY}/Selection"


class RegionEditDetectionsToRegionCrops(smart_nodes.BoundingBoxCropBatch):
    CATEGORY = f"{ROOT_CATEGORY}/Selection"


class RegionEditImageGridWindows(smart_nodes.ImageGridWindows):
    CATEGORY = f"{ROOT_CATEGORY}/Selection"


class RegionEditBoundedImageSizePlan(face_nodes.SemanticObjectEditorSizePlan):
    CATEGORY = f"{ROOT_CATEGORY}/Geometry"


class RegionEditExactIntegerDownscale(face_nodes.FaceLocalExactIntegerDownscale):
    CATEGORY = f"{ROOT_CATEGORY}/Geometry"


class RegionEditFaceContextCrop(face_nodes.FaceLocalFaceContextSquareCrop):
    CATEGORY = f"{ROOT_CATEGORY}/Geometry"


class RegionEditTilePlanItem(smart_nodes.MaskRegionTileAtIndex):
    CATEGORY = f"{ROOT_CATEGORY}/Geometry"


class RegionEditMaskSetCompose(face_nodes.FaceLocalComposeMask):
    CATEGORY = f"{ROOT_CATEGORY}/Mask"


class RegionEditInteractiveMaskCompose(smart_nodes.MaskUnionManualProtect):
    CATEGORY = f"{ROOT_CATEGORY}/Mask"


class RegionEditAlignedDifferenceMask(face_nodes.SemanticObjectDifferenceMaskExact):
    CATEGORY = f"{ROOT_CATEGORY}/Mask"

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
        source = face_nodes._image_batch(source_local)
        generated = face_nodes._image_batch(generated_local)
        if int(source.shape[0]) != 1 or int(generated.shape[0]) != 1:
            raise ValueError(
                "aligned difference requires exactly one source image and one generated image"
            )
        if tuple(source.shape) != tuple(generated.shape):
            raise ValueError(
                "source_local and generated_local must have identical dimensions"
            )
        expected_height, expected_width = int(source.shape[1]), int(source.shape[2])
        for name, value in (
            ("target_core_mask", target_core_mask),
            ("protection_mask", protection_mask),
            ("editable_support_mask", editable_support_mask),
        ):
            batch, height, width = _single_mask_geometry(value, name)
            if batch != 1:
                raise ValueError(f"{name} must contain exactly one mask")
            if (height, width) != (expected_height, expected_width):
                raise ValueError(
                    f"{name} dimensions must exactly match source_local and "
                    f"generated_local; received {width}x{height}, expected "
                    f"{expected_width}x{expected_height}"
                )
        return super().build(
            source_local=source_local,
            generated_local=generated_local,
            target_core_mask=target_core_mask,
            protection_mask=protection_mask,
            editable_support_mask=editable_support_mask,
            threshold_level=threshold_level,
            difference_expand=difference_expand,
            contract_version=contract_version,
        )


class RegionEditMaskEditorCanvas(face_nodes.FaceLocalMaskEditorCanvas):
    CATEGORY = f"{ROOT_CATEGORY}/Mask"


class RegionEditMaskEditorApply:
    """Apply one complete MaskEditor result with an optional core lock.

    Geometry, placeholder handling, support clipping, and manual-delta outputs
    are identical in both modes.  ``lock_mandatory_core`` changes only whether
    the mandatory core is restored into the final mask.
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
            },
            "optional": {
                "mandatory_core_mask": ("MASK",),
                "lock_mandatory_core": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "keep mandatory core",
                        "label_off": "allow complete erase",
                    },
                ),
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
    CATEGORY = f"{ROOT_CATEGORY}/Mask"

    def correct(
        self,
        difference_image,
        automatic_mask,
        edited_mask,
        processing_support_mask,
        support_threshold=0.001,
        mandatory_core_mask=None,
        lock_mandatory_core=False,
        apply_manual_editor=True,
    ):
        difference = face_nodes._image_batch(difference_image).to(
            device="cpu", dtype=torch.float32
        )
        height, width = int(difference.shape[1]), int(difference.shape[2])
        expected_size = (height, width)

        for name, value in (
            ("automatic_mask", automatic_mask),
            ("processing_support_mask", processing_support_mask),
        ):
            batch, mask_height, mask_width = _single_mask_geometry(value, name)
            if batch != 1:
                raise ValueError(f"{name} must contain exactly one mask")
            if (mask_height, mask_width) != expected_size:
                raise ValueError(
                    f"{name} dimensions must exactly match the correction canvas; "
                    f"received {mask_width}x{mask_height}, expected {width}x{height}"
                )

        edited_batch, edited_height, edited_width = _single_mask_geometry(
            edited_mask, "edited_mask"
        )
        if edited_batch != 1:
            raise ValueError("edited_mask must contain exactly one mask")
        raw_editor_size = (edited_height, edited_width)
        manual_editor_requested = bool(apply_manual_editor)
        placeholder_fallback = bool(
            manual_editor_requested
            and raw_editor_size == (64, 64)
            and raw_editor_size != expected_size
        )
        if (
            manual_editor_requested
            and raw_editor_size != expected_size
            and not placeholder_fallback
        ):
            raise ValueError(
                "edited_mask dimensions do not match the correction canvas: "
                f"received {edited_width}x{edited_height}, "
                f"expected {width}x{height}"
            )

        lock_core = bool(lock_mandatory_core)
        if lock_core and mandatory_core_mask is None:
            raise ValueError(
                "mandatory_core_mask is required when lock_mandatory_core is enabled"
            )

        threshold = float(support_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")

        automatic = face_nodes._mask_batch(
            automatic_mask, height, width
        ).to(device="cpu", dtype=torch.float32)
        support = face_nodes._mask_batch(
            processing_support_mask, height, width
        ).to(device="cpu", dtype=torch.float32)
        support_selected = support > threshold
        if not torch.any(support_selected):
            raise ValueError("processing_support_mask is empty")

        if not manual_editor_requested or placeholder_fallback:
            edited = automatic.clone()
            manual_editor_applied = False
        else:
            edited = face_nodes._mask_batch(
                edited_mask, height, width
            ).to(device="cpu", dtype=torch.float32)
            manual_editor_applied = True

        automatic_in_support = torch.where(
            support_selected, automatic, torch.zeros_like(automatic)
        )
        edited_in_support = torch.where(
            support_selected, edited, torch.zeros_like(edited)
        )
        raw_add = torch.clamp(
            edited_in_support - automatic_in_support, 0.0, 1.0
        )
        raw_erase = torch.clamp(
            automatic_in_support - edited_in_support, 0.0, 1.0
        )
        manual_add = raw_add.masked_fill(raw_add < threshold, 0.0)
        manual_erase = raw_erase.masked_fill(raw_erase < threshold, 0.0)

        protected_core_restore = torch.zeros_like(automatic)
        final_mask = edited_in_support.clone()
        core_lock_passed = True
        if lock_core:
            core_batch, core_height, core_width = _single_mask_geometry(
                mandatory_core_mask, "mandatory_core_mask"
            )
            if core_batch != 1:
                raise ValueError("mandatory_core_mask must contain exactly one mask")
            if (core_height, core_width) != expected_size:
                raise ValueError(
                    "mandatory_core_mask dimensions must exactly match the "
                    f"correction canvas; received {core_width}x{core_height}, "
                    f"expected {width}x{height}"
                )
            core = face_nodes._mask_batch(
                mandatory_core_mask, height, width
            ).to(device="cpu", dtype=torch.float32)
            core_selected = core > threshold
            if torch.any(core_selected & ~support_selected):
                raise ValueError(
                    "mandatory_core_mask extends outside processing support"
                )
            protected_core_restore = torch.clamp(
                core - edited_in_support, 0.0, 1.0
            )
            protected_core_restore = torch.where(
                support_selected,
                protected_core_restore,
                torch.zeros_like(protected_core_restore),
            )
            final_mask = torch.maximum(final_mask, core)
            final_mask = torch.where(
                support_selected, final_mask, torch.zeros_like(final_mask)
            )
            final_mask[core_selected] = 1.0
            core_lock_passed = bool(
                torch.all(final_mask[core_selected] == 1.0)
            )

        final_mask[final_mask < threshold] = 0.0
        outside_final_zero = bool(
            torch.all(final_mask[~support_selected] == 0.0)
        )
        outside_add_zero = bool(
            torch.all(manual_add[~support_selected] == 0.0)
        )
        outside_erase_zero = bool(
            torch.all(manual_erase[~support_selected] == 0.0)
        )

        preview = difference.clone()
        layers = (
            (final_mask, (1.0, 0.12, 0.08), 0.34),
            (manual_add, (0.05, 1.0, 0.15), 0.72),
            (manual_erase, (1.0, 0.05, 0.85), 0.72),
            (protected_core_restore, (1.0, 0.90, 0.05), 0.72),
        )
        for layer, color_values, opacity in layers:
            weight = torch.clamp(
                layer.unsqueeze(-1) * float(opacity), 0.0, float(opacity)
            )
            color = torch.tensor(color_values, dtype=torch.float32).view(
                1, 1, 1, 3
            )
            preview = preview * (1.0 - weight) + color * weight

        gate_passed = bool(
            outside_final_zero
            and outside_add_zero
            and outside_erase_zero
            and core_lock_passed
        )
        report = {
            "algorithm": "region-edit-complete-manual-mask-correction-v1",
            "lock_mandatory_core": lock_core,
            "manual_editor_requested": manual_editor_requested,
            "editor_canvas_initialized": raw_editor_size == expected_size,
            "previewbridge_placeholder_fallback": placeholder_fallback,
            "manual_editor_applied": manual_editor_applied,
            "manual_editor_bypassed": not manual_editor_applied,
            "raw_editor_size_hw": [edited_height, edited_width],
            "expected_editor_size_hw": [height, width],
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
            "protected_core_restore_pixels": int(
                torch.count_nonzero(protected_core_restore > threshold).item()
            ),
            "final_support_pixels": int(
                torch.count_nonzero(final_mask > threshold).item()
            ),
            "outside_processing_support": {
                "final_is_zero": outside_final_zero,
                "manual_add_is_zero": outside_add_zero,
                "manual_erase_is_zero": outside_erase_zero,
            },
            "mandatory_core_lock_passed": core_lock_passed,
            "preview_legend": {
                "red": "final replacement mask",
                "green": "manual add",
                "magenta": "manual erase",
                "yellow": "mandatory core restored by lock",
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


class RegionEditMaskContentGate(face_nodes.FaceLocalNonEmptyMaskGate):
    CATEGORY = f"{ROOT_CATEGORY}/Mask"


class RegionEditFaceAdaptiveDifferenceMask(
    face_nodes.FaceLocalAdaptiveDifferenceMask
):
    CATEGORY = f"{ROOT_CATEGORY}/Face"


class RegionEditFaceThresholdDifferenceMask(
    face_nodes.FaceLocalThresholdDifferenceMask
):
    CATEGORY = f"{ROOT_CATEGORY}/Face"


def _single_mask_geometry(value, name: str) -> tuple[int, int, int]:
    """Return ``(batch, height, width)`` without resizing or merging masks."""

    shape = tuple(value.shape) if hasattr(value, "shape") else tuple(np.asarray(value).shape)
    if len(shape) == 2:
        return 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    if len(shape) == 4 and int(shape[-1]) == 1:
        return int(shape[0]), int(shape[1]), int(shape[2])
    if len(shape) == 4 and int(shape[1]) == 1:
        return int(shape[0]), int(shape[2]), int(shape[3])
    raise ValueError(
        f"{name} must be HW, BHW, BHW1, or B1HW; received shape {shape}"
    )


class RegionEditBoundaryFeather4Side:
    """Build a four-sided crop-perimeter alpha without scene semantics.

    In orientation-default mode, portrait uses top/bottom 10 and left/right 5;
    landscape uses left/right 10 and top/bottom 5; square uses 10 on all sides.
    In physical-side mode, all four visible percentages are used literally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_mask": ("MASK",),
                "original_image": ("IMAGE",),
                "x": ("INT", {"forceInput": True}),
                "y": ("INT", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "use_orientation_defaults": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "10% long sides / 5% short sides",
                        "label_off": "use four physical side values",
                    },
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
                "feather_at_original_image_boundary": (
                    "BOOLEAN",
                    {"default": False},
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("feathered_alpha", "four_side_ramp", "report_json")
    FUNCTION = "feather"
    CATEGORY = f"{ROOT_CATEGORY}/Mask"

    @staticmethod
    def _side_ramp(length: int, width: int, reverse: bool = False) -> np.ndarray:
        ramp = np.ones((length,), dtype=np.float32)
        if width > 0:
            usable = min(int(width), int(length))
            values = (
                np.array([0.0], dtype=np.float32)
                if usable == 1
                else np.linspace(0.0, 1.0, usable, dtype=np.float32)
            )
            if reverse:
                ramp[-usable:] = values[::-1]
            else:
                ramp[:usable] = values
        return ramp

    def feather(
        self,
        local_mask,
        original_image,
        x,
        y,
        width,
        height,
        use_orientation_defaults=True,
        top_feather_percent=10.0,
        bottom_feather_percent=10.0,
        left_feather_percent=5.0,
        right_feather_percent=5.0,
        feather_at_original_image_boundary=False,
    ):
        source = face_nodes._image_batch(original_image).to(
            device="cpu", dtype=torch.float32
        )
        full_height, full_width = int(source.shape[1]), int(source.shape[2])
        px, py, crop_width, crop_height = int(x), int(y), int(width), int(height)
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError("width and height must be positive")
        if px < 0 or py < 0 or px + crop_width > full_width or py + crop_height > full_height:
            raise ValueError(
                "crop coordinates must stay inside original_image for four-side feathering"
            )

        mask_batch, mask_height, mask_width = _single_mask_geometry(
            local_mask, "local_mask"
        )
        if mask_batch != 1:
            raise ValueError("four-side feather requires mask batch size 1")
        if (mask_height, mask_width) != (crop_height, crop_width):
            raise ValueError(
                "local_mask dimensions must exactly match width and height; "
                f"received {mask_width}x{mask_height}, expected {crop_width}x{crop_height}"
            )
        mask = face_nodes._mask_batch(local_mask, crop_height, crop_width).to(
            device="cpu", dtype=torch.float32
        )

        entered = (
            float(top_feather_percent),
            float(bottom_feather_percent),
            float(left_feather_percent),
            float(right_feather_percent),
        )
        if bool(use_orientation_defaults) and crop_width > crop_height:
            effective = (5.0, 5.0, 10.0, 10.0)
            interpretation = "orientation-default-landscape"
        elif bool(use_orientation_defaults) and crop_width == crop_height:
            effective = (10.0, 10.0, 10.0, 10.0)
            interpretation = "orientation-default-square"
        elif bool(use_orientation_defaults):
            effective = (10.0, 10.0, 5.0, 5.0)
            interpretation = "orientation-default-portrait"
        else:
            effective = entered
            interpretation = "literal-physical-sides"

        top_percent, bottom_percent, left_percent, right_percent = effective
        top_pixels = int(round(crop_height * top_percent / 100.0))
        bottom_pixels = int(round(crop_height * bottom_percent / 100.0))
        left_pixels = int(round(crop_width * left_percent / 100.0))
        right_pixels = int(round(crop_width * right_percent / 100.0))

        if not bool(feather_at_original_image_boundary):
            if py == 0:
                top_pixels = 0
            if py + crop_height == full_height:
                bottom_pixels = 0
            if px == 0:
                left_pixels = 0
            if px + crop_width == full_width:
                right_pixels = 0

        vertical = np.minimum(
            self._side_ramp(crop_height, top_pixels),
            self._side_ramp(crop_height, bottom_pixels, reverse=True),
        )
        horizontal = np.minimum(
            self._side_ramp(crop_width, left_pixels),
            self._side_ramp(crop_width, right_pixels, reverse=True),
        )
        ramp_np = np.minimum(vertical[:, None], horizontal[None, :])
        ramp = torch.from_numpy(ramp_np).unsqueeze(0)
        feathered = torch.clamp(mask * ramp, 0.0, 1.0)
        report = {
            "operation": "four-side-crop-perimeter-feather",
            "crop_xywh": [px, py, crop_width, crop_height],
            "original_size": [full_width, full_height],
            "entered_percent": {
                "top": entered[0],
                "bottom": entered[1],
                "left": entered[2],
                "right": entered[3],
            },
            "effective_percent": {
                "top": top_percent,
                "bottom": bottom_percent,
                "left": left_percent,
                "right": right_percent,
            },
            "effective_pixels": {
                "top": top_pixels,
                "bottom": bottom_pixels,
                "left": left_pixels,
                "right": right_pixels,
            },
            "interpretation": interpretation,
            "use_orientation_defaults": bool(use_orientation_defaults),
            "feather_at_original_image_boundary": bool(
                feather_at_original_image_boundary
            ),
        }
        return (
            feathered,
            ramp,
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class RegionEditProtectedOuterBoundaryFeather(
    face_nodes.FaceLocalOuterBoundaryFeather
):
    CATEGORY = f"{ROOT_CATEGORY}/Mask"


class RegionEditTilePlanner(smart_nodes.MaskRegionTilePlanner):
    CATEGORY = f"{ROOT_CATEGORY}/Tile"


class RegionEditBasicTileBatchPrepare(smart_nodes.MaskRegionTileBatch):
    CATEGORY = f"{ROOT_CATEGORY}/Tile"


class RegionEditTileBatchPrepare(smart_nodes.MaskRegionTileBatchControlled):
    CATEGORY = f"{ROOT_CATEGORY}/Tile"


class RegionEditTileMerge(smart_nodes.MaskRegionWeightedMerge):
    CATEGORY = f"{ROOT_CATEGORY}/Tile"


class RegionEditWindowMaskMerge(smart_nodes.MaskGridMerge):
    CATEGORY = f"{ROOT_CATEGORY}/Tile"


class RegionEditTileControlPreset(smart_nodes.LocalEditTileControls):
    CATEGORY = f"{ROOT_CATEGORY}/Workflow"


class RegionEditStrictCoordinateComposite(face_nodes.FaceLocalStrictComposite):
    CATEGORY = f"{ROOT_CATEGORY}/Composite"


class RegionEditWideSupportStrictComposite(
    face_nodes.SemanticObjectStrictCompositeExact
):
    CATEGORY = f"{ROOT_CATEGORY}/Composite"


class RegionEditMaskedGlobalLabMatch(face_nodes.FaceLocalContextColorHarmonize):
    CATEGORY = f"{ROOT_CATEGORY}/Color & Detail"


class RegionEditMaskedSpatialColorFieldMatch(
    face_nodes.SemanticObjectContextColorMatch
):
    CATEGORY = f"{ROOT_CATEGORY}/Color & Detail"


class RegionEditMaskedHighFrequencyTransfer(
    face_nodes.FaceLocalOriginalHighFrequencyTransfer
):
    CATEGORY = f"{ROOT_CATEGORY}/Color & Detail"


class RegionEditSkinMicrotextureSynthesis(
    face_nodes.FaceLocalSyntheticSkinMicrotexture
):
    CATEGORY = f"{ROOT_CATEGORY}/Color & Detail"


class RegionEditProtectedDetailHarmonizer(face_nodes.FaceLocalDetailHarmonizer):
    CATEGORY = f"{ROOT_CATEGORY}/Color & Detail"


class RegionEditFaceIdentityConditioningMask(
    face_nodes.FaceLocalIdentityConditioningMask
):
    CATEGORY = f"{ROOT_CATEGORY}/Face"


class RegionEditFaceEyeMaterialRestore(face_nodes.FaceLocalSourceEyeMaterialRestore):
    CATEGORY = f"{ROOT_CATEGORY}/Face"


class RegionEditFaceStructureDelta(face_nodes.FaceLocalStructureDelta):
    CATEGORY = f"{ROOT_CATEGORY}/Face"


class RegionEditFaceSemanticCrop(face_nodes.FaceLocalSemanticCrop):
    CATEGORY = f"{ROOT_CATEGORY}/Face"


class RegionEditReferenceSourceSelector(face_nodes.FaceLocalImageSourceSelector):
    CATEGORY = f"{ROOT_CATEGORY}/Workflow"


class RegionEditPromptRouteSelector(face_nodes.FaceLocalRouteSelector):
    CATEGORY = f"{ROOT_CATEGORY}/Workflow"


class RegionEditFacePromptContract(face_nodes.FaceLocalPromptContract):
    CATEGORY = f"{ROOT_CATEGORY}/Text"


class RegionEditPortraitPromptQualityGate(face_nodes.FaceLocalPromptQualityGate):
    CATEGORY = f"{ROOT_CATEGORY}/Text"


class RegionEditReplacementPromptContract(
    face_nodes.SemanticObjectReplacementPromptContract
):
    CATEGORY = f"{ROOT_CATEGORY}/Text"


class RegionEditPreservationPromptBuilder(smart_nodes.AppendPreservationPrompt):
    CATEGORY = f"{ROOT_CATEGORY}/Text"


class RegionEditRequiredOfflineEnglish(smart_nodes.SAMPromptAutoEnglish):
    CATEGORY = f"{ROOT_CATEGORY}/Text"


class RegionEditOptionalOfflineEnglish(face_nodes.SemanticObjectOptionalSAMPrompt):
    CATEGORY = f"{ROOT_CATEGORY}/Text"


CANONICAL_NODE_CLASS_MAPPINGS = {
    "RegionEditFaceSelect": RegionEditFaceSelect,
    "RegionEditDetectionsToRegionCrops": RegionEditDetectionsToRegionCrops,
    "RegionEditImageGridWindows": RegionEditImageGridWindows,
    "RegionEditBoundedImageSizePlan": RegionEditBoundedImageSizePlan,
    "RegionEditExactIntegerDownscale": RegionEditExactIntegerDownscale,
    "RegionEditFaceContextCrop": RegionEditFaceContextCrop,
    "RegionEditTilePlanItem": RegionEditTilePlanItem,
    "RegionEditMaskSetCompose": RegionEditMaskSetCompose,
    "RegionEditInteractiveMaskCompose": RegionEditInteractiveMaskCompose,
    "RegionEditAlignedDifferenceMask": RegionEditAlignedDifferenceMask,
    "RegionEditMaskEditorCanvas": RegionEditMaskEditorCanvas,
    "RegionEditMaskEditorApply": RegionEditMaskEditorApply,
    "RegionEditMaskContentGate": RegionEditMaskContentGate,
    "RegionEditFaceAdaptiveDifferenceMask": RegionEditFaceAdaptiveDifferenceMask,
    "RegionEditFaceThresholdDifferenceMask": RegionEditFaceThresholdDifferenceMask,
    "RegionEditBoundaryFeather4Side": RegionEditBoundaryFeather4Side,
    "RegionEditProtectedOuterBoundaryFeather": RegionEditProtectedOuterBoundaryFeather,
    "RegionEditTilePlanner": RegionEditTilePlanner,
    "RegionEditBasicTileBatchPrepare": RegionEditBasicTileBatchPrepare,
    "RegionEditTileBatchPrepare": RegionEditTileBatchPrepare,
    "RegionEditTileMerge": RegionEditTileMerge,
    "RegionEditWindowMaskMerge": RegionEditWindowMaskMerge,
    "RegionEditTileControlPreset": RegionEditTileControlPreset,
    "RegionEditStrictCoordinateComposite": RegionEditStrictCoordinateComposite,
    "RegionEditWideSupportStrictComposite": RegionEditWideSupportStrictComposite,
    "RegionEditMaskedGlobalLabMatch": RegionEditMaskedGlobalLabMatch,
    "RegionEditMaskedSpatialColorFieldMatch": RegionEditMaskedSpatialColorFieldMatch,
    "RegionEditMaskedHighFrequencyTransfer": RegionEditMaskedHighFrequencyTransfer,
    "RegionEditSkinMicrotextureSynthesis": RegionEditSkinMicrotextureSynthesis,
    "RegionEditProtectedDetailHarmonizer": RegionEditProtectedDetailHarmonizer,
    "RegionEditFaceIdentityConditioningMask": RegionEditFaceIdentityConditioningMask,
    "RegionEditFaceEyeMaterialRestore": RegionEditFaceEyeMaterialRestore,
    "RegionEditFaceStructureDelta": RegionEditFaceStructureDelta,
    "RegionEditFaceSemanticCrop": RegionEditFaceSemanticCrop,
    "RegionEditReferenceSourceSelector": RegionEditReferenceSourceSelector,
    "RegionEditPromptRouteSelector": RegionEditPromptRouteSelector,
    "RegionEditFacePromptContract": RegionEditFacePromptContract,
    "RegionEditPortraitPromptQualityGate": RegionEditPortraitPromptQualityGate,
    "RegionEditReplacementPromptContract": RegionEditReplacementPromptContract,
    "RegionEditPreservationPromptBuilder": RegionEditPreservationPromptBuilder,
    "RegionEditRequiredOfflineEnglish": RegionEditRequiredOfflineEnglish,
    "RegionEditOptionalOfflineEnglish": RegionEditOptionalOfflineEnglish,
}


CANONICAL_NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionEditFaceSelect": "图像分区 · 选择人脸",
    "RegionEditDetectionsToRegionCrops": "图像分区 · 检测框转区域裁块",
    "RegionEditImageGridWindows": "图像分区 · 图像扫描窗口",
    "RegionEditBoundedImageSizePlan": "图像分区 · 像素区间尺寸规划",
    "RegionEditExactIntegerDownscale": "图像分区 · 16 对齐整数倍缩小",
    "RegionEditFaceContextCrop": "图像分区 · 人脸上下文裁块",
    "RegionEditTilePlanItem": "图像分区 · 读取分块计划项",
    "RegionEditMaskSetCompose": "图像分区 · 遮罩集合合成",
    "RegionEditInteractiveMaskCompose": "图像分区 · 自动／手工／保护遮罩合成",
    "RegionEditAlignedDifferenceMask": "图像分区 · 对齐图差异遮罩",
    "RegionEditMaskEditorCanvas": "图像分区 · 手涂遮罩画布",
    "RegionEditMaskEditorApply": "图像分区 · 应用手涂遮罩",
    "RegionEditMaskContentGate": "图像分区 · 遮罩有效性检查",
    "RegionEditFaceAdaptiveDifferenceMask": "图像分区 · 人脸自适应差异与核心",
    "RegionEditFaceThresholdDifferenceMask": "图像分区 · 人脸阈值差异与核心",
    "RegionEditBoundaryFeather4Side": "图像分区 · 四边羽化",
    "RegionEditProtectedOuterBoundaryFeather": "图像分区 · 外边界羽化与核心保护",
    "RegionEditTilePlanner": "图像分区 · 多区域分块规划",
    "RegionEditBasicTileBatchPrepare": "图像分区 · 基础批量区域裁块",
    "RegionEditTileBatchPrepare": "图像分区 · 批量准备区域分块",
    "RegionEditTileMerge": "图像分区 · 多分块归一化合并",
    "RegionEditWindowMaskMerge": "图像分区 · 扫描窗口遮罩合并",
    "RegionEditTileControlPreset": "图像分区 · 分块参数档位",
    "RegionEditStrictCoordinateComposite": "图像分区 · 严格坐标合成",
    "RegionEditWideSupportStrictComposite": "图像分区 · 宽支持区域严格合成",
    "RegionEditMaskedGlobalLabMatch": "图像分区 · 遮罩区域 Lab 匹配",
    "RegionEditMaskedSpatialColorFieldMatch": "图像分区 · 空间低频颜色匹配",
    "RegionEditMaskedHighFrequencyTransfer": "图像分区 · 遮罩高频迁移",
    "RegionEditSkinMicrotextureSynthesis": "图像分区 · 皮肤微纹理合成",
    "RegionEditProtectedDetailHarmonizer": "图像分区 · 受保护细节一致化",
    "RegionEditFaceIdentityConditioningMask": "图像分区 · 人脸身份参考遮罩",
    "RegionEditFaceEyeMaterialRestore": "图像分区 · 眼内材质恢复",
    "RegionEditFaceStructureDelta": "图像分区 · 人脸结构差异",
    "RegionEditFaceSemanticCrop": "图像分区 · 人脸语义裁块",
    "RegionEditReferenceSourceSelector": "图像分区 · 参考图来源选择",
    "RegionEditPromptRouteSelector": "图像分区 · Prompt 路线选择",
    "RegionEditFacePromptContract": "图像分区 · 人脸编辑 Prompt 合约",
    "RegionEditPortraitPromptQualityGate": "图像分区 · 人像 Prompt 质量检查",
    "RegionEditReplacementPromptContract": "图像分区 · 替换编辑 Prompt 合约",
    "RegionEditPreservationPromptBuilder": "图像分区 · 保留约束 Prompt 构建",
    "RegionEditRequiredOfflineEnglish": "图像分区 · 必填离线英译",
    "RegionEditOptionalOfflineEnglish": "图像分区 · 可选离线英译",
}
