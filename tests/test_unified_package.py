from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy_node_contracts.json"
WORKFLOW_LEGACY_IDS_FIXTURE = (
    ROOT / "tests" / "fixtures" / "formal_workflow_legacy_ids.json"
)
LEGACY_INPUT_IS_LIST = {
    "UniversalMaskGridMergeExact": True,
    "UniversalRegionWeightedMergeExact": True,
}


def load_plugin():
    package_name = "region_edit_toolkit_test_plugin"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLUGIN = load_plugin()
MIGRATION_MAP = importlib.import_module(
    f"{PLUGIN.__name__}.region_edit_toolkit.migration_map"
).LEGACY_TO_CANONICAL_NODE_IDS


def schema_hash(cls) -> str:
    normalized = json.loads(
        json.dumps(
            cls.INPUT_TYPES(),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class UnifiedRegistryTests(unittest.TestCase):
    def test_registry_contains_only_42_pure_v2_ids(self):
        self.assertEqual(len(PLUGIN.NODE_CLASS_MAPPINGS), 42)
        self.assertTrue(
            all(key.startswith("RegionEdit") for key in PLUGIN.NODE_CLASS_MAPPINGS)
        )
        self.assertEqual(
            set(PLUGIN.NODE_CLASS_MAPPINGS),
            set(PLUGIN.NODE_DISPLAY_NAME_MAPPINGS),
        )

    def test_all_registered_classes_have_a_callable_contract(self):
        for node_id, cls in PLUGIN.NODE_CLASS_MAPPINGS.items():
            with self.subTest(node_id=node_id):
                self.assertTrue(hasattr(cls, "INPUT_TYPES"))
                self.assertTrue(hasattr(cls, "RETURN_TYPES"))
                self.assertTrue(hasattr(cls, "FUNCTION"))
                self.assertTrue(hasattr(cls, "CATEGORY"))
                self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
                schema = cls.INPUT_TYPES()
                self.assertIsInstance(schema, dict)
                self.assertEqual(
                    len(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)),
                    len(cls.RETURN_TYPES),
                )

    def test_retired_ids_are_documented_but_not_registered(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["legacy_node_count"], 43)
        expected_ids = set(fixture["nodes"])
        self.assertEqual(set(MIGRATION_MAP), expected_ids)
        self.assertTrue(expected_ids.isdisjoint(PLUGIN.NODE_CLASS_MAPPINGS))
        self.assertTrue(
            set(MIGRATION_MAP.values()).issubset(PLUGIN.NODE_CLASS_MAPPINGS)
        )

    def test_one_to_one_migrations_preserve_the_serialized_contract(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        consolidated_manual_ids = {
            "FaceLocalManualAdaptiveMaskCorrection",
            "SemanticObjectManualMaskCorrection",
        }
        for node_id, expected in fixture["nodes"].items():
            if node_id in consolidated_manual_ids:
                continue
            cls = PLUGIN.NODE_CLASS_MAPPINGS[MIGRATION_MAP[node_id]]
            schema = cls.INPUT_TYPES()
            with self.subTest(node_id=node_id):
                self.assertEqual(cls.FUNCTION, expected["function"])
                self.assertEqual(list(cls.RETURN_TYPES), expected["return_types"])
                self.assertEqual(
                    list(getattr(cls, "RETURN_NAMES", ())),
                    expected["return_names"],
                )
                self.assertEqual(
                    list(getattr(cls, "OUTPUT_IS_LIST", ())),
                    expected["output_is_list"],
                )
                self.assertEqual(
                    bool(getattr(cls, "INPUT_IS_LIST", False)),
                    LEGACY_INPUT_IS_LIST.get(node_id, False),
                )
                self.assertFalse(bool(getattr(cls, "OUTPUT_NODE", False)))
                self.assertEqual(
                    list(schema.get("required", {})),
                    expected["required_inputs"],
                )
                self.assertEqual(
                    list(schema.get("optional", {})),
                    expected["optional_inputs"],
                )
                self.assertEqual(
                    list(schema.get("hidden", {})),
                    expected["hidden_inputs"],
                )
                self.assertEqual(schema_hash(cls), expected["schema_sha256"])

    def test_consolidated_mask_editor_apply_covers_both_retired_contracts(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        schema = cls.INPUT_TYPES()
        self.assertEqual(
            list(schema["required"]),
            [
                "difference_image",
                "automatic_mask",
                "edited_mask",
                "processing_support_mask",
                "support_threshold",
            ],
        )
        self.assertEqual(
            list(schema["optional"]),
            [
                "mandatory_core_mask",
                "lock_mandatory_core",
                "apply_manual_editor",
            ],
        )

    def test_formal_workflow_retired_id_closure_has_a_v2_mapping(self):
        fixture = json.loads(
            WORKFLOW_LEGACY_IDS_FIXTURE.read_text(encoding="utf-8")
        )
        required = set()
        for workflow_name, node_ids in fixture["workflows"].items():
            with self.subTest(workflow=workflow_name):
                self.assertTrue(set(node_ids).issubset(MIGRATION_MAP))
                self.assertTrue(
                    {
                        MIGRATION_MAP[node_id] for node_id in node_ids
                    }.issubset(PLUGIN.NODE_CLASS_MAPPINGS)
                )
            required.update(node_ids)
        self.assertEqual(len(required), fixture["unique_legacy_id_count"])


class CanonicalPrimitiveTests(unittest.TestCase):
    def test_replacement_prompt_contract_isolates_reference_background(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS[
            "RegionEditReplacementPromptContract"
        ]
        instruction = "把图1的耳机包换成图2的鼠标。"

        klein, gpt_text_only, gpt_reference = cls().build(instruction)

        self.assertIn(instruction, klein)
        self.assertIn("Do not copy Image 2's background", klein)
        self.assertIn("Preserve every other", klein)
        self.assertNotIn("Image 2", gpt_text_only)
        self.assertIn("Do not copy Image 2's background", gpt_reference)

    def test_replacement_prompt_contract_rejects_an_empty_instruction(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS[
            "RegionEditReplacementPromptContract"
        ]

        with self.assertRaises(ValueError):
            cls().build("   ")

    def test_mask_editor_apply_without_core_lock_allows_complete_erase(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 8, 8), dtype=torch.float32)
        automatic[:, 2:6, 2:6] = 1.0
        edited = torch.zeros_like(automatic)
        support = torch.ones_like(automatic)

        final_mask, _, erase, _, _ = cls().correct(
            image,
            automatic,
            edited,
            support,
            lock_mandatory_core=False,
        )
        self.assertEqual(int(torch.count_nonzero(final_mask)), 0)
        self.assertEqual(int(torch.count_nonzero(erase)), 16)

    def test_mask_editor_apply_with_core_lock_restores_the_core(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 8, 8), dtype=torch.float32)
        automatic[:, 2:6, 2:6] = 1.0
        edited = torch.zeros_like(automatic)
        support = torch.ones_like(automatic)
        core = torch.zeros_like(automatic)
        core[:, 3:5, 3:5] = 1.0

        final_mask, _, _, _, _ = cls().correct(
            image,
            automatic,
            edited,
            support,
            lock_mandatory_core=True,
            mandatory_core_mask=core,
        )
        self.assertEqual(int(torch.count_nonzero(final_mask)), 4)
        self.assertTrue(torch.all(final_mask[:, 3:5, 3:5] == 1.0))

    def test_mask_editor_apply_rejects_a_missing_locked_core(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        mask = torch.zeros((1, 8, 8), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "mandatory_core_mask"):
            cls().correct(
                image,
                mask,
                mask,
                torch.ones_like(mask),
                lock_mandatory_core=True,
            )

    def test_mask_editor_apply_can_bypass_an_initialized_editor_and_keep_core(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 8, 8), dtype=torch.float32)
        automatic[:, 2:6, 2:6] = 1.0
        edited = torch.ones_like(automatic)
        support = torch.ones_like(automatic)
        core = torch.zeros_like(automatic)
        core[:, 3:5, 3:5] = 1.0

        final_mask, add, erase, _, report_json = cls().correct(
            image,
            automatic,
            edited,
            support,
            mandatory_core_mask=core,
            lock_mandatory_core=True,
            apply_manual_editor=False,
        )
        self.assertTrue(torch.equal(final_mask, automatic))
        self.assertEqual(int(torch.count_nonzero(add)), 0)
        self.assertEqual(int(torch.count_nonzero(erase)), 0)
        report = json.loads(report_json)
        self.assertFalse(report["manual_editor_requested"])
        self.assertTrue(report["manual_editor_bypassed"])

    def test_mask_editor_apply_rejects_non_placeholder_size_in_both_modes(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 8, 8), dtype=torch.float32)
        edited = torch.zeros((1, 7, 8), dtype=torch.float32)
        support = torch.ones_like(automatic)
        core = torch.zeros_like(automatic)
        for lock_core in (False, True):
            with self.subTest(lock_mandatory_core=lock_core):
                with self.assertRaisesRegex(ValueError, "edited_mask dimensions"):
                    cls().correct(
                        image,
                        automatic,
                        edited,
                        support,
                        lock_mandatory_core=lock_core,
                        mandatory_core_mask=core,
                    )

    def test_mask_editor_apply_uses_64px_placeholder_in_both_modes(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.zeros((1, 8, 8), dtype=torch.float32)
        automatic[:, 2:6, 2:6] = 1.0
        placeholder = torch.zeros((1, 64, 64), dtype=torch.float32)
        support = torch.ones_like(automatic)
        core = torch.zeros_like(automatic)
        core[:, 3:5, 3:5] = 1.0

        for lock_core in (False, True):
            with self.subTest(lock_mandatory_core=lock_core):
                final_mask, add, erase, _, report_json = cls().correct(
                    image,
                    automatic,
                    placeholder,
                    support,
                    lock_mandatory_core=lock_core,
                    mandatory_core_mask=core,
                )
                self.assertTrue(torch.equal(final_mask, automatic))
                self.assertEqual(int(torch.count_nonzero(add)), 0)
                self.assertEqual(int(torch.count_nonzero(erase)), 0)
                self.assertTrue(
                    json.loads(report_json)["previewbridge_placeholder_fallback"]
                )

    def test_mask_editor_manual_deltas_are_zero_outside_support_in_both_modes(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditMaskEditorApply"]
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        automatic = torch.ones((1, 8, 8), dtype=torch.float32)
        edited = torch.zeros_like(automatic)
        support = torch.zeros_like(automatic)
        support[:, 2:6, 2:6] = 1.0
        core = torch.zeros_like(automatic)
        core[:, 3:5, 3:5] = 1.0
        outside_support = support == 0

        for lock_core in (False, True):
            with self.subTest(lock_mandatory_core=lock_core):
                final_mask, add, erase, _, _ = cls().correct(
                    image,
                    automatic,
                    edited,
                    support,
                    lock_mandatory_core=lock_core,
                    mandatory_core_mask=core,
                )
                self.assertTrue(torch.all(final_mask[outside_support] == 0.0))
                self.assertTrue(torch.all(add[outside_support] == 0.0))
                self.assertTrue(torch.all(erase[outside_support] == 0.0))

    def test_four_side_feather_uses_orientation_defaults(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditBoundaryFeather4Side"]
        original = torch.zeros((1, 200, 200, 3), dtype=torch.float32)
        portrait_mask = torch.ones((1, 100, 50), dtype=torch.float32)
        alpha, ramp, report_json = cls().feather(
            portrait_mask,
            original,
            x=50,
            y=50,
            width=50,
            height=100,
        )
        report = json.loads(report_json)
        self.assertEqual(
            report["effective_percent"],
            {"top": 10.0, "bottom": 10.0, "left": 5.0, "right": 5.0},
        )
        self.assertEqual(tuple(alpha.shape), (1, 100, 50))
        self.assertEqual(tuple(ramp.shape), (1, 100, 50))
        self.assertEqual(float(ramp[0, 0, 25]), 0.0)
        self.assertEqual(float(ramp[0, 50, 0]), 0.0)
        self.assertEqual(float(ramp[0, 50, 25]), 1.0)

        landscape_mask = torch.ones((1, 50, 100), dtype=torch.float32)
        _, _, landscape_report_json = cls().feather(
            landscape_mask,
            original,
            x=50,
            y=50,
            width=100,
            height=50,
        )
        landscape_report = json.loads(landscape_report_json)
        self.assertEqual(
            landscape_report["effective_percent"],
            {"top": 5.0, "bottom": 5.0, "left": 10.0, "right": 10.0},
        )

    def test_four_side_feather_can_use_literal_physical_side_values(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditBoundaryFeather4Side"]
        original = torch.zeros((1, 200, 200, 3), dtype=torch.float32)
        landscape_mask = torch.ones((1, 50, 100), dtype=torch.float32)
        _, _, report_json = cls().feather(
            landscape_mask,
            original,
            x=50,
            y=50,
            width=100,
            height=50,
            use_orientation_defaults=False,
            top_feather_percent=10.0,
            bottom_feather_percent=10.0,
            left_feather_percent=5.0,
            right_feather_percent=5.0,
        )
        report = json.loads(report_json)
        self.assertEqual(
            report["effective_percent"],
            {"top": 10.0, "bottom": 10.0, "left": 5.0, "right": 5.0},
        )
        self.assertEqual(report["interpretation"], "literal-physical-sides")

    def test_four_side_feather_rejects_out_of_bounds_coordinates(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditBoundaryFeather4Side"]
        original = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
        local = torch.ones((1, 40, 40), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "inside original_image"):
            cls().feather(
                local,
                original,
                x=80,
                y=80,
                width=40,
                height=40,
            )

    def test_four_side_feather_rejects_a_mismatched_mask_instead_of_resizing(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditBoundaryFeather4Side"]
        original = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
        local = torch.ones((1, 39, 40), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            cls().feather(
                local,
                original,
                x=20,
                y=20,
                width=40,
                height=40,
            )

    def test_aligned_difference_accepts_one_pair_and_rejects_image_batches(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditAlignedDifferenceMask"]
        source = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        generated = source.clone()
        generated[:, 3:5, 3:5, :] = 1.0
        target = torch.zeros((1, 8, 8), dtype=torch.float32)
        target[:, 3:5, 3:5] = 1.0
        protection = torch.zeros_like(target)
        support = torch.ones_like(target)

        automatic, bounded, _, _ = cls().build(
            source,
            generated,
            target,
            protection,
            support,
            threshold_level=7,
            difference_expand=0,
        )
        self.assertEqual(int(torch.count_nonzero(automatic)), 4)
        self.assertEqual(int(torch.count_nonzero(bounded)), 4)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            cls().build(
                source.repeat(2, 1, 1, 1),
                generated.repeat(2, 1, 1, 1),
                target,
                protection,
                support,
                threshold_level=7,
                difference_expand=0,
            )

    def test_aligned_difference_rejects_mismatched_mask_dimensions(self):
        cls = PLUGIN.NODE_CLASS_MAPPINGS["RegionEditAlignedDifferenceMask"]
        source = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        generated = source.clone()
        target = torch.ones((1, 8, 7), dtype=torch.float32)
        protection = torch.zeros((1, 8, 8), dtype=torch.float32)
        support = torch.ones((1, 8, 8), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "target_core_mask dimensions"):
            cls().build(
                source,
                generated,
                target,
                protection,
                support,
                threshold_level=7,
                difference_expand=0,
            )


if __name__ == "__main__":
    unittest.main()
