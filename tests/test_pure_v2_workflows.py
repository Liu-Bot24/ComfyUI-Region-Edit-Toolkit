from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PAIRS = {
    "ComfyUI-Smart-Removal.json": "ComfyUI-Smart-Removal-RegionEdit-v2.json",
    "FaceLocal-Dual-Reference-Final.json": (
        "FaceLocal-Dual-Reference-Final-RegionEdit-v2.json"
    ),
    "ComfyUI-Semantic-Object-Replacement.json": (
        "ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json"
    ),
}
EXPECTED_WORKFLOW_SUMMARIES = {
    "ComfyUI-Smart-Removal-RegionEdit-v2.json": {
        "nodes": 72,
        "graphs": 7,
        "links": 187,
        "region_edit_nodes": 6,
        "layout_sha256": (
            "f995a13fdfaeeeef93a129b93b12ed0f249e8e6a6c7248278d2a3b84e2e2a621"
        ),
        "semantic_links_sha256": (
            "24205f7ecbe74bb5d0ce757e6b97997fff9fcccf85e1337cb0fa2437632b24b5"
        ),
    },
    "FaceLocal-Dual-Reference-Final-RegionEdit-v2.json": {
        "nodes": 215,
        "graphs": 16,
        "links": 441,
        "region_edit_nodes": 16,
        "layout_sha256": (
            "9a503d601d6c3ef9f915d06b8765fd9cbfd0870a1e5f9737b56ecd187634febd"
        ),
        "semantic_links_sha256": (
            "da940368e32da391d7636f904eb08dc9e20e9e44a4e9c47f0b730069ca94d714"
        ),
    },
    "ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json": {
        "nodes": 140,
        "graphs": 11,
        "links": 375,
        "region_edit_nodes": 8,
        "layout_sha256": (
            "465dbbc36524d9252ebc308569fcd9bbe5430fd138c897adf5c9751305087a90"
        ),
        "semantic_links_sha256": (
            "d01bbbed0076852700bd81ebffff00dbbb0e64f9530ccdfc02b44e249ab2855d"
        ),
    },
}


def load_plugin():
    package_name = "region_edit_toolkit_workflow_test_plugin"
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


def walk_nodes(value):
    if isinstance(value, dict):
        if (
            isinstance(value.get("type"), str)
            and isinstance(value.get("id"), int)
            and isinstance(value.get("inputs"), list)
            and isinstance(value.get("outputs"), list)
        ):
            yield value
        for child in value.values():
            yield from walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_nodes(child)


def walk_graphs(value, path="root"):
    if isinstance(value, dict):
        if isinstance(value.get("nodes"), list) and isinstance(
            value.get("links"), list
        ):
            yield path, value
        for key, child in value.items():
            yield from walk_graphs(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_graphs(child, f"{path}[{index}]")


def link_fields(link):
    if isinstance(link, dict):
        return (
            link.get("id"),
            link.get("origin_id"),
            link.get("origin_slot"),
            link.get("target_id"),
            link.get("target_slot"),
            link.get("type"),
        )
    if isinstance(link, list) and len(link) >= 6:
        return tuple(link[:6])
    raise AssertionError(f"unsupported workflow link record: {link!r}")


def serialized_link_contract_issues(graph):
    node_by_id = {
        node["id"]: node
        for node in graph["nodes"]
        if isinstance(node, dict) and isinstance(node.get("id"), int)
    }
    boundary_inputs = graph.get("inputs", [])
    boundary_outputs = graph.get("outputs", [])
    issues = []
    for link in graph["links"]:
        (
            link_id,
            origin_id,
            origin_slot,
            target_id,
            target_slot,
            link_type,
        ) = link_fields(link)

        if origin_id == -10:
            if not 0 <= origin_slot < len(boundary_inputs):
                issues.append((link_id, "missing boundary input"))
                origin = None
            else:
                origin = boundary_inputs[origin_slot]
                if link_id not in origin.get("linkIds", []):
                    issues.append((link_id, "boundary input linkIds mismatch"))
        else:
            origin = node_by_id.get(origin_id, {}).get("outputs", [])[origin_slot]

        if target_id == -20:
            if not 0 <= target_slot < len(boundary_outputs):
                issues.append((link_id, "missing boundary output"))
                target = None
            else:
                target = boundary_outputs[target_slot]
                if link_id not in target.get("linkIds", []):
                    issues.append((link_id, "boundary output linkIds mismatch"))
        else:
            target = node_by_id.get(target_id, {}).get("inputs", [])[target_slot]

        for endpoint_name, endpoint in (("origin", origin), ("target", target)):
            if endpoint is None:
                continue
            endpoint_type = endpoint.get("type")
            if (
                link_type != "*"
                and endpoint_type != "*"
                and link_type != endpoint_type
            ):
                issues.append(
                    (
                        link_id,
                        f"{endpoint_name} type {endpoint_type} != {link_type}",
                    )
                )
    return issues


def layout_signature(workflow):
    signature = []
    for graph_path, graph in walk_graphs(workflow):
        for node_index, node in enumerate(graph["nodes"]):
            if not (
                isinstance(node, dict)
                and isinstance(node.get("id"), int)
                and isinstance(node.get("type"), str)
            ):
                continue
            signature.append(
                (
                    graph_path,
                    node_index,
                    node["id"],
                    {
                        key: copy.deepcopy(node.get(key))
                        for key in ("pos", "size", "flags", "order", "mode")
                    },
                )
            )
    return signature


def semantic_link_signature(workflow):
    signature = []
    for graph_path, graph in walk_graphs(workflow):
        node_by_id = {
            node["id"]: node
            for node in graph["nodes"]
            if isinstance(node, dict) and isinstance(node.get("id"), int)
        }
        for link_index, link in enumerate(graph["links"]):
            (
                link_id,
                origin_id,
                origin_slot,
                target_id,
                target_slot,
                link_type,
            ) = link_fields(link)
            origin_node = node_by_id.get(origin_id)
            target_node = node_by_id.get(target_id)
            origin_name = origin_slot
            target_name = target_slot
            if origin_node is not None:
                outputs = origin_node.get("outputs", [])
                if 0 <= origin_slot < len(outputs):
                    origin_name = outputs[origin_slot].get("name", origin_slot)
            if target_node is not None:
                inputs = target_node.get("inputs", [])
                if 0 <= target_slot < len(inputs):
                    target_name = inputs[target_slot].get("name", target_slot)
            signature.append(
                (
                    graph_path,
                    link_index,
                    link_id,
                    origin_id,
                    origin_name,
                    target_id,
                    target_name,
                    link_type,
                )
            )
    return signature


def input_types_for(node_class):
    schema = node_class.INPUT_TYPES()
    result = {}
    for section in ("required", "optional", "hidden"):
        for name, spec in schema.get(section, {}).items():
            value = spec[0] if isinstance(spec, tuple) else spec
            result[name] = "COMBO" if isinstance(value, (list, tuple)) else value
    return result


def signature_sha256(value):
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PureV2WorkflowTests(unittest.TestCase):
    def test_public_object_example_uses_frontend_saved_inline_controls_without_gpt(self):
        workflow = json.loads(
            (
                ROOT
                / "workflows"
                / "examples"
                / "RegionEdit-Object-Replacement-Example.json"
            ).read_text(encoding="utf-8")
        )

        root_nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(len(root_nodes), 26)
        self.assertEqual(len(workflow["links"]), 62)
        self.assertEqual(len(workflow["definitions"]["subgraphs"]), 10)
        self.assertEqual(workflow.get("revision"), 1)

        self.assertFalse(
            any(
                node["type"] in {"PrimitiveString", "PrimitiveStringMultiline"}
                for node in workflow["nodes"]
            )
        )
        forbidden_gpt_types = {
            "GPTImageBridgeOAuthProvider",
            "GPTImageBridgeAPIProvider",
            "GPTImageBridgeReferenceList",
            "GPTImageBridgeEdit",
        }
        self.assertTrue(
            forbidden_gpt_types.isdisjoint(
                {node["type"] for node in walk_nodes(workflow)}
            )
        )

        sam = root_nodes[200]
        sam_inputs = {item["name"]: item for item in sam["inputs"]}
        self.assertIsNone(sam_inputs["处理对象"]["link"])
        self.assertIsNone(sam_inputs["保护对象"]["link"])
        self.assertEqual(
            sam["widgets_values"],
            ["杯子", "人", False, False],
        )
        self.assertNotIn("proxyWidgets", sam.get("properties", {}))

        generation_root = root_nodes[205]
        generation_inputs = {
            item["name"]: item for item in generation_root["inputs"]
        }
        self.assertNotIn("使用GPT", generation_inputs)
        self.assertNotIn("GPT Provider", generation_inputs)
        self.assertIsNone(generation_inputs["替换要求"]["link"])
        self.assertEqual(
            generation_root["widgets_values"][1],
            "把图 1 中的这杯咖啡换成图 2 中的米饭，保留勺子。",
        )
        self.assertNotIn("proxyWidgets", generation_root.get("properties", {}))

        self.assertEqual(
            root_nodes[220]["widgets_values"],
            [False, False, False, True, True],
        )
        self.assertEqual(
            root_nodes[1]["widgets_values"][0],
            "u_1mx30qjvv0-online-education-10399630.jpg",
        )
        self.assertEqual(
            root_nodes[2]["widgets_values"][0],
            "istockphoto-1224135793-612x612.jpg",
        )

        contracts = [
            node
            for node in walk_nodes(workflow)
            if node.get("type") == "RegionEditReplacementPromptContract"
        ]
        self.assertEqual(len(contracts), 1)

        contract = contracts[0]
        self.assertIsNotNone(contract["inputs"][0]["link"])
        self.assertEqual(contract["outputs"][0]["name"], "klein_prompt")
        self.assertEqual(len(contract["outputs"][0]["links"]), 1)

        clip_nodes = [
            node
            for node in walk_nodes(workflow)
            if node.get("type") == "CLIPTextEncode"
        ]
        self.assertTrue(
            any(
                node["inputs"][1]["link"]
                == contract["outputs"][0]["links"][0]
                for node in clip_nodes
            )
        )

    def test_public_examples_have_closed_serialized_links(self):
        for workflow_path in (ROOT / "workflows" / "examples").glob("*.json"):
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            for graph_path, graph in walk_graphs(workflow):
                self.assertEqual(
                    serialized_link_contract_issues(graph),
                    [],
                    f"{workflow_path.name}: {graph_path}",
                )
                node_by_id = {
                    node["id"]: node
                    for node in graph["nodes"]
                    if isinstance(node, dict) and isinstance(node.get("id"), int)
                }
                links = {}
                for link in graph["links"]:
                    fields = link_fields(link)
                    link_id, origin_id, origin_slot, target_id, target_slot, _ = (
                        fields
                    )
                    with self.subTest(
                        workflow=workflow_path.name,
                        graph=graph_path,
                        link=link_id,
                    ):
                        self.assertNotIn(link_id, links)
                        links[link_id] = fields
                        self.assertTrue(origin_id in node_by_id or origin_id == -10)
                        self.assertTrue(target_id in node_by_id or target_id == -20)
                        if origin_id in node_by_id:
                            self.assertIn(
                                link_id,
                                node_by_id[origin_id]["outputs"][origin_slot].get(
                                    "links"
                                )
                                or [],
                            )
                        if target_id in node_by_id:
                            self.assertEqual(
                                node_by_id[target_id]["inputs"][target_slot].get(
                                    "link"
                                ),
                                link_id,
                            )

                for node in node_by_id.values():
                    for item in node.get("inputs", []):
                        if item.get("link") is not None:
                            self.assertIn(item["link"], links)
                    for item in node.get("outputs", []):
                        for link_id in item.get("links") or []:
                            self.assertIn(link_id, links)

    def test_public_examples_identify_region_edit_nodes_for_manager(self):
        canonical_types = set(PLUGIN.NODE_CLASS_MAPPINGS)
        for workflow_path in (ROOT / "workflows" / "examples").glob("*.json"):
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            package_nodes = [
                node
                for node in walk_nodes(workflow)
                if node.get("type") in canonical_types
            ]
            self.assertTrue(package_nodes, workflow_path.name)
            for node in package_nodes:
                with self.subTest(
                    workflow=workflow_path.name,
                    node_id=node["id"],
                    node_type=node["type"],
                ):
                    properties = node.get("properties", {})
                    self.assertEqual(
                        properties.get("cnr_id"),
                        "native-region-tile-planner-merge",
                    )
                    self.assertEqual(properties.get("ver"), "0.2.0")

    def test_public_link_contract_check_rejects_known_port_shifts(self):
        workflow = json.loads(
            (
                ROOT
                / "workflows"
                / "examples"
                / "RegionEdit-Object-Replacement-Example.json"
            ).read_text(encoding="utf-8")
        )
        generation = next(
            subgraph
            for subgraph in workflow["definitions"]["subgraphs"]
            if subgraph["id"] == "28b3e999-4800-5f56-9d14-cd777d921ced"
        )
        self.assertEqual(serialized_link_contract_issues(generation), [])

        shifted_boundary = copy.deepcopy(generation)
        next(
            link for link in shifted_boundary["links"] if link["id"] == 57
        )["origin_slot"] = 6
        self.assertIn(
            (57, "boundary input linkIds mismatch"),
            serialized_link_contract_issues(shifted_boundary),
        )

        mask_on_boolean = copy.deepcopy(generation)
        mask_link = next(
            link for link in mask_on_boolean["links"] if link["id"] == 50
        )
        mask_link["target_id"] = 427
        mask_link["target_slot"] = 2
        self.assertIn(
            (50, "target type BOOLEAN != MASK"),
            serialized_link_contract_issues(mask_on_boolean),
        )

    def test_reviewed_workflow_structure_signatures_remain_stable(self):
        retired = set(MIGRATION_MAP)
        for output_name, expected in EXPECTED_WORKFLOW_SUMMARIES.items():
            with self.subTest(workflow=output_name):
                migrated = json.loads(
                    (ROOT / "workflows" / output_name).read_text(encoding="utf-8")
                )
                graphs = list(walk_graphs(migrated))
                nodes = list(walk_nodes(migrated))
                self.assertEqual(
                    len(nodes),
                    expected["nodes"],
                )
                self.assertEqual(
                    len(graphs),
                    expected["graphs"],
                )
                self.assertEqual(
                    sum(len(graph["links"]) for _, graph in graphs),
                    expected["links"],
                )
                self.assertEqual(
                    sum(node["type"].startswith("RegionEdit") for node in nodes),
                    expected["region_edit_nodes"],
                )
                self.assertEqual(
                    signature_sha256(layout_signature(migrated)),
                    expected["layout_sha256"],
                )
                self.assertEqual(
                    signature_sha256(semantic_link_signature(migrated)),
                    expected["semantic_links_sha256"],
                )
                self.assertTrue(
                    retired.isdisjoint(
                        {node["type"] for node in nodes}
                    )
                )

    def test_every_published_workflow_uses_only_current_node_ids(self):
        retired = set(MIGRATION_MAP)
        for workflow_path in (ROOT / "workflows").glob("*.json"):
            with self.subTest(workflow=workflow_path.name):
                workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
                self.assertTrue(
                    retired.isdisjoint(
                        {node["type"] for node in walk_nodes(workflow)}
                    )
                )

    def test_every_migrated_region_edit_node_matches_its_serialized_ports(self):
        for output_name in WORKFLOW_PAIRS.values():
            workflow = json.loads(
                (ROOT / "workflows" / output_name).read_text(encoding="utf-8")
            )
            for node in walk_nodes(workflow):
                node_type = node["type"]
                if not node_type.startswith("RegionEdit"):
                    continue
                with self.subTest(workflow=output_name, node_id=node["id"]):
                    self.assertIn(node_type, PLUGIN.NODE_CLASS_MAPPINGS)
                    node_class = PLUGIN.NODE_CLASS_MAPPINGS[node_type]
                    valid_inputs = input_types_for(node_class)
                    for item in node.get("inputs", []):
                        self.assertIn(item["name"], valid_inputs)
                        self.assertEqual(item["type"], valid_inputs[item["name"]])
                    expected_names = list(
                        getattr(node_class, "RETURN_NAMES", node_class.RETURN_TYPES)
                    )
                    expected_types = list(node_class.RETURN_TYPES)
                    self.assertEqual(
                        [item["name"] for item in node.get("outputs", [])],
                        expected_names,
                    )
                    self.assertEqual(
                        [item["type"] for item in node.get("outputs", [])],
                        expected_types,
                    )
                    self.assertEqual(
                        node.get("properties", {}).get("Node name for S&R"),
                        node_type,
                    )

    def test_every_serialized_link_matches_both_node_side_references(self):
        for output_name in WORKFLOW_PAIRS.values():
            workflow = json.loads(
                (ROOT / "workflows" / output_name).read_text(encoding="utf-8")
            )
            for graph_path, graph in walk_graphs(workflow):
                node_by_id = {
                    node["id"]: node
                    for node in graph["nodes"]
                    if isinstance(node, dict) and isinstance(node.get("id"), int)
                }
                graph_links = {}
                for link in graph["links"]:
                    (
                        link_id,
                        origin_id,
                        origin_slot,
                        target_id,
                        target_slot,
                        _,
                    ) = link_fields(link)
                    with self.subTest(
                        workflow=output_name,
                        graph=graph_path,
                        link=link_id,
                    ):
                        self.assertNotIn(link_id, graph_links)
                        graph_links[link_id] = link
                        self.assertTrue(
                            origin_id in node_by_id or origin_id == -10
                        )
                        self.assertTrue(
                            target_id in node_by_id or target_id == -20
                        )
                        if origin_id in node_by_id:
                            origin_outputs = node_by_id[origin_id].get(
                                "outputs", []
                            )
                            self.assertGreater(len(origin_outputs), origin_slot)
                            self.assertIn(
                                link_id,
                                origin_outputs[origin_slot].get("links") or [],
                            )
                        if target_id in node_by_id:
                            target_inputs = node_by_id[target_id].get(
                                "inputs", []
                            )
                            self.assertGreater(len(target_inputs), target_slot)
                            self.assertEqual(
                                target_inputs[target_slot].get("link"),
                                link_id,
                            )

                for node in node_by_id.values():
                    for input_slot, item in enumerate(node.get("inputs", [])):
                        link_id = item.get("link")
                        if link_id is None:
                            continue
                        with self.subTest(
                            workflow=output_name,
                            graph=graph_path,
                            node=node["id"],
                            input_slot=input_slot,
                        ):
                            self.assertIn(link_id, graph_links)
                            self.assertEqual(
                                link_fields(graph_links[link_id])[3:5],
                                (node["id"], input_slot),
                            )
                    for output_slot, item in enumerate(node.get("outputs", [])):
                        for link_id in item.get("links") or []:
                            with self.subTest(
                                workflow=output_name,
                                graph=graph_path,
                                node=node["id"],
                                output_slot=output_slot,
                            ):
                                self.assertIn(link_id, graph_links)
                                self.assertEqual(
                                    link_fields(graph_links[link_id])[1:3],
                                    (node["id"], output_slot),
                                )

    def test_manual_mask_nodes_preserve_their_original_modes(self):
        face = json.loads(
            (
                ROOT
                / "workflows"
                / "FaceLocal-Dual-Reference-Final-RegionEdit-v2.json"
            ).read_text(encoding="utf-8")
        )
        face_nodes = {node["id"]: node for node in walk_nodes(face)}
        self.assertEqual(face_nodes[53]["widgets_values"], [0.001, True, False])
        self.assertEqual(face_nodes[372]["widgets_values"], [0.001, True, True])

        object_workflow = json.loads(
            (
                ROOT
                / "workflows"
                / "ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json"
            ).read_text(encoding="utf-8")
        )
        object_nodes = {node["id"]: node for node in walk_nodes(object_workflow)}
        self.assertEqual(
            object_nodes[512]["widgets_values"],
            [0.001, False, True],
        )

    def test_repository_workflow_copies_do_not_store_api_credentials(self):
        for output_name in WORKFLOW_PAIRS.values():
            workflow = json.loads(
                (ROOT / "workflows" / output_name).read_text(encoding="utf-8")
            )
            for node in walk_nodes(workflow):
                if node.get("type") != "GPTImageBridgeAPIProvider":
                    continue
                input_names = [
                    item.get("name") for item in node.get("inputs", [])
                ]
                api_key_index = input_names.index("api_key")
                self.assertEqual(node["widgets_values"][api_key_index], "")


if __name__ == "__main__":
    unittest.main()
