# 退役节点 ID 映射表

本表只用于把旧工作流迁移为纯新版工作流。左侧 ID 不会被当前插件注册。

## Smart / Region Tile

| 退役 ID | 纯新版 ID |
|---|---|
| `MaskRegionTilePlannerExact` | `RegionEditTilePlanner` |
| `MaskRegionTileAtIndexExact` | `RegionEditTilePlanItem` |
| `UniversalImageGridWindowsExact` | `RegionEditImageGridWindows` |
| `UniversalBoundingBoxCropBatchExact` | `RegionEditDetectionsToRegionCrops` |
| `UniversalMaskGridMergeExact` | `RegionEditWindowMaskMerge` |
| `UniversalMaskUnionManualProtectExact` | `RegionEditInteractiveMaskCompose` |
| `UniversalRegionTileBatchExact` | `RegionEditBasicTileBatchPrepare` |
| `UniversalRegionTileBatchControlledExact` | `RegionEditTileBatchPrepare` |
| `UniversalLocalEditTileControls` | `RegionEditTileControlPreset` |
| `UniversalRegionWeightedMergeExact` | `RegionEditTileMerge` |
| `UniversalAppendPreservationPrompt` | `RegionEditPreservationPromptBuilder` |
| `UniversalSAMPromptAutoEnglish` | `RegionEditRequiredOfflineEnglish` |

## Face Local

| 退役 ID | 纯新版 ID |
|---|---|
| `FaceLocalAdaptiveDifferenceMask` | `RegionEditFaceAdaptiveDifferenceMask` |
| `FaceLocalComposeMask` | `RegionEditMaskSetCompose` |
| `FaceLocalContextColorHarmonize` | `RegionEditMaskedGlobalLabMatch` |
| `FaceLocalDetailHarmonizer` | `RegionEditProtectedDetailHarmonizer` |
| `FaceLocalExactIntegerDownscale` | `RegionEditExactIntegerDownscale` |
| `FaceLocalFaceContextSquareCrop` | `RegionEditFaceContextCrop` |
| `FaceLocalIdentityConditioningMask` | `RegionEditFaceIdentityConditioningMask` |
| `FaceLocalImageSourceSelector` | `RegionEditReferenceSourceSelector` |
| `FaceLocalManualAdaptiveMaskCorrection` | `RegionEditMaskEditorApply` |
| `FaceLocalMaskEditorCanvas` | `RegionEditMaskEditorCanvas` |
| `FaceLocalNonEmptyMaskGate` | `RegionEditMaskContentGate` |
| `FaceLocalOriginalHighFrequencyTransfer` | `RegionEditMaskedHighFrequencyTransfer` |
| `FaceLocalOuterBoundaryFeather` | `RegionEditProtectedOuterBoundaryFeather` |
| `FaceLocalPromptContract` | `RegionEditFacePromptContract` |
| `FaceLocalPromptQualityGate` | `RegionEditPortraitPromptQualityGate` |
| `FaceLocalRouteSelector` | `RegionEditPromptRouteSelector` |
| `FaceLocalSelectFace` | `RegionEditFaceSelect` |
| `FaceLocalSemanticCrop` | `RegionEditFaceSemanticCrop` |
| `FaceLocalSourceEyeMaterialRestore` | `RegionEditFaceEyeMaterialRestore` |
| `FaceLocalStrictComposite` | `RegionEditStrictCoordinateComposite` |
| `FaceLocalStructureDelta` | `RegionEditFaceStructureDelta` |
| `FaceLocalSyntheticSkinMicrotexture` | `RegionEditSkinMicrotextureSynthesis` |
| `FaceLocalThresholdDifferenceMask` | `RegionEditFaceThresholdDifferenceMask` |

## Semantic Object

| 退役 ID | 纯新版 ID |
|---|---|
| `SemanticObjectContextColorMatch` | `RegionEditMaskedSpatialColorFieldMatch` |
| `SemanticObjectDifferenceMaskExact` | `RegionEditAlignedDifferenceMask` |
| `SemanticObjectEditorSizePlan` | `RegionEditBoundedImageSizePlan` |
| `SemanticObjectManualMaskCorrection` | `RegionEditMaskEditorApply` |
| `SemanticObjectMaskEditorCanvas` | `RegionEditMaskEditorCanvas` |
| `SemanticObjectOptionalSAMPrompt` | `RegionEditOptionalOfflineEnglish` |
| `SemanticObjectReplacementPromptContract` | `RegionEditReplacementPromptContract` |
| `SemanticObjectStrictCompositeExact` | `RegionEditWideSupportStrictComposite` |

## 两项需要显式模式迁移的节点

`RegionEditMaskEditorApply` 合并了 Face 与 Semantic Object 两个旧应用节点。迁移时不能只替换 `type`：

- Face 工作流：`lock_mandatory_core=true`；`apply_manual_editor` 继承原开关。
- Semantic Object 工作流：`lock_mandatory_core=false`；`apply_manual_editor=true`。

仓库迁移脚本已经按上述规则生成三份工作流副本。
