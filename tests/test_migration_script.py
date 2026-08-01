from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "migrate_formal_workflows_to_pure_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "region_edit_pure_v2_migration_script",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MIGRATION_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION_SCRIPT)


def minimal_workflow(node_type: str):
    return {
        "nodes": [
            {
                "id": 1,
                "type": node_type,
                "pos": [10, 20],
                "size": [300, 200],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {
                    "Node name for S&R": node_type,
                    "cnr_id": "retired-package",
                    "ver": "1.0.0",
                },
                "widgets_values": [],
            }
        ],
        "links": [],
    }


class MigrationScriptTests(unittest.TestCase):
    def test_explicit_source_directory_generates_all_three_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_dir = temporary_path / "source"
            output_dir = temporary_path / "output"
            source_dir.mkdir()
            for source_name in MIGRATION_SCRIPT.WORKFLOWS:
                (source_dir / source_name).write_text(
                    json.dumps(minimal_workflow("MaskRegionTilePlannerExact")),
                    encoding="utf-8",
                )

            report = MIGRATION_SCRIPT.migrate_workflow_set(
                source_dir,
                output_dir,
            )

            self.assertEqual(
                set(report),
                set(MIGRATION_SCRIPT.WORKFLOWS.values()),
            )
            for output_name in MIGRATION_SCRIPT.WORKFLOWS.values():
                migrated = json.loads(
                    (output_dir / output_name).read_text(encoding="utf-8")
                )
                node = migrated["nodes"][0]
                self.assertEqual(node["type"], "RegionEditTilePlanner")
                self.assertEqual(
                    node["properties"]["Node name for S&R"],
                    "RegionEditTilePlanner",
                )
                self.assertNotIn("cnr_id", node["properties"])
                self.assertNotIn("ver", node["properties"])

    def test_missing_source_aborts_before_creating_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_dir = temporary_path / "source"
            output_dir = temporary_path / "output"
            source_dir.mkdir()
            first_source = next(iter(MIGRATION_SCRIPT.WORKFLOWS))
            (source_dir / first_source).write_text(
                json.dumps(minimal_workflow("MaskRegionTilePlannerExact")),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                MIGRATION_SCRIPT.migrate_workflow_set(
                    source_dir,
                    output_dir,
                )

            self.assertFalse(output_dir.exists())

    def test_face_manual_mask_links_follow_the_reordered_named_inputs(self):
        node = {
            "id": 53,
            "type": "FaceLocalManualAdaptiveMaskCorrection",
            "pos": [10, 20],
            "size": [300, 200],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [
                {"name": "difference_image", "type": "IMAGE", "link": 1},
                {"name": "automatic_mask", "type": "MASK", "link": 2},
                {"name": "edited_mask", "type": "MASK", "link": 3},
                {
                    "name": "support_threshold",
                    "type": "FLOAT",
                    "link": None,
                    "widget": {"name": "support_threshold"},
                },
                {
                    "name": "mandatory_core_mask",
                    "type": "MASK",
                    "link": 5,
                },
                {
                    "name": "processing_support_mask",
                    "type": "MASK",
                    "link": 4,
                },
            ],
            "outputs": [],
            "properties": {
                "Node name for S&R": (
                    "FaceLocalManualAdaptiveMaskCorrection"
                ),
            },
            "widgets_values": [0.001, False],
        }
        links = [
            [1, -10, 0, 53, 0, "IMAGE"],
            [2, -10, 1, 53, 1, "MASK"],
            [3, -10, 2, 53, 2, "MASK"],
            [5, -10, 3, 53, 4, "MASK"],
            [4, -10, 4, 53, 5, "MASK"],
        ]
        workflow = {"nodes": [node], "links": links}

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_path = temporary_path / "source.json"
            output_path = temporary_path / "output.json"
            source_path.write_text(json.dumps(workflow), encoding="utf-8")

            MIGRATION_SCRIPT.migrate(source_path, output_path)
            migrated = json.loads(output_path.read_text(encoding="utf-8"))

        migrated_node = migrated["nodes"][0]
        self.assertEqual(
            migrated_node["type"],
            "RegionEditMaskEditorApply",
        )
        self.assertEqual(
            [item["name"] for item in migrated_node["inputs"]],
            [
                "difference_image",
                "automatic_mask",
                "edited_mask",
                "processing_support_mask",
                "support_threshold",
                "mandatory_core_mask",
                "lock_mandatory_core",
                "apply_manual_editor",
            ],
        )
        self.assertEqual(migrated_node["widgets_values"], [0.001, True, False])
        links_by_id = {link[0]: link for link in migrated["links"]}
        self.assertEqual(links_by_id[4][4], 3)
        self.assertEqual(links_by_id[5][4], 5)


if __name__ == "__main__":
    unittest.main()
