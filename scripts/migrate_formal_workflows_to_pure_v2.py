from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from region_edit_toolkit.migration_map import LEGACY_TO_CANONICAL_NODE_IDS


DEFAULT_OUTPUT_DIR = ROOT / "workflows"
WORKFLOWS = {
    "ComfyUI-Smart-Removal.json": "ComfyUI-Smart-Removal-RegionEdit-v2.json",
    "FaceLocal-Dual-Reference-Final.json": (
        "FaceLocal-Dual-Reference-Final-RegionEdit-v2.json"
    ),
    "ComfyUI-Semantic-Object-Replacement.json": (
        "ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json"
    ),
}


def _boolean_input(name: str):
    return {
        "label": name,
        "localized_name": name,
        "name": name,
        "shape": 7,
        "type": "BOOLEAN",
        "widget": {"name": name},
        "link": None,
    }


def _optional_mask_input(name: str):
    return {
        "label": name,
        "localized_name": name,
        "name": name,
        "shape": 7,
        "type": "MASK",
        "link": None,
    }


def _link_fields(link):
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
    raise ValueError(f"unsupported workflow link record: {link!r}")


def _set_link_target_slot(link, target_slot: int):
    if isinstance(link, dict):
        link["target_slot"] = target_slot
    elif isinstance(link, list) and len(link) >= 6:
        link[4] = target_slot
    else:
        raise ValueError(f"unsupported workflow link record: {link!r}")


def _migrate_manual_mask_apply(node: dict, old_type: str, graph_links: list):
    inputs = {item["name"]: copy.deepcopy(item) for item in node["inputs"]}
    widgets = list(node.get("widgets_values", []))
    support_threshold = widgets[0] if widgets else 0.001

    mandatory_core = inputs.get("mandatory_core_mask")
    if mandatory_core is None:
        mandatory_core = _optional_mask_input("mandatory_core_mask")
    else:
        mandatory_core["shape"] = 7

    if old_type == "FaceLocalManualAdaptiveMaskCorrection":
        apply_manual_editor = bool(widgets[1]) if len(widgets) > 1 else True
        lock_mandatory_core = True
    elif old_type == "SemanticObjectManualMaskCorrection":
        apply_manual_editor = True
        lock_mandatory_core = False
    else:
        raise ValueError(f"unexpected manual-mask source type: {old_type}")

    node["inputs"] = [
        inputs["difference_image"],
        inputs["automatic_mask"],
        inputs["edited_mask"],
        inputs["processing_support_mask"],
        inputs["support_threshold"],
        mandatory_core,
        _boolean_input("lock_mandatory_core"),
        _boolean_input("apply_manual_editor"),
    ]
    node["widgets_values"] = [
        support_threshold,
        lock_mandatory_core,
        apply_manual_editor,
    ]

    linked_input_slots = {
        item["link"]: index
        for index, item in enumerate(node["inputs"])
        if item.get("link") is not None
    }
    updated_link_ids = set()
    for link in graph_links:
        link_id, _, _, target_id, _, _ = _link_fields(link)
        if target_id != node["id"] or link_id not in linked_input_slots:
            continue
        _set_link_target_slot(link, linked_input_slots[link_id])
        updated_link_ids.add(link_id)
    missing_link_ids = set(linked_input_slots).difference(updated_link_ids)
    if missing_link_ids:
        raise RuntimeError(
            f"node {node['id']} is missing serialized links for inputs: "
            f"{sorted(missing_link_ids)}"
        )


def _walk_nodes(value):
    if isinstance(value, dict):
        if (
            isinstance(value.get("type"), str)
            and isinstance(value.get("id"), int)
            and isinstance(value.get("inputs"), list)
            and isinstance(value.get("outputs"), list)
        ):
            yield value
        for child in value.values():
            yield from _walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nodes(child)


def _walk_graphs(value, path="root"):
    if isinstance(value, dict):
        if isinstance(value.get("nodes"), list) and isinstance(
            value.get("links"), list
        ):
            yield path, value
        for key, child in value.items():
            yield from _walk_graphs(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_graphs(child, f"{path}[{index}]")


def _redact_serialized_secrets(workflow: dict):
    """Remove credentials from repository workflow copies.

    Formal user workflows remain untouched.  Only the generated copies are
    sanitized so a saved API provider cannot leak its credential into Git.
    """

    redacted = 0
    for node in _walk_nodes(workflow):
        if node.get("type") != "GPTImageBridgeAPIProvider":
            continue
        inputs = list(node.get("inputs", []))
        widgets = list(node.get("widgets_values", []))
        api_key_indexes = [
            index
            for index, item in enumerate(inputs)
            if item.get("name") == "api_key"
        ]
        if len(api_key_indexes) != 1:
            raise RuntimeError(
                "GPTImageBridgeAPIProvider must expose exactly one api_key widget"
            )
        api_key_index = api_key_indexes[0]
        if api_key_index >= len(widgets):
            raise RuntimeError(
                "GPTImageBridgeAPIProvider api_key widget value is missing"
            )
        if widgets[api_key_index]:
            redacted += 1
        widgets[api_key_index] = ""
        node["widgets_values"] = widgets
    return redacted


def _layout_signature(workflow: dict):
    signature = []
    for graph_path, graph in _walk_graphs(workflow):
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


def _semantic_link_signature(workflow: dict):
    signature = []
    for graph_path, graph in _walk_graphs(workflow):
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
            ) = _link_fields(link)
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


def migrate(source_path: Path, output_path: Path):
    workflow = json.loads(source_path.read_text(encoding="utf-8"))
    original_layout = _layout_signature(workflow)
    original_links = _semantic_link_signature(workflow)
    changed = []

    for _, graph in _walk_graphs(workflow):
        for node in graph["nodes"]:
            if not isinstance(node, dict):
                continue
            old_type = node.get("type")
            new_type = LEGACY_TO_CANONICAL_NODE_IDS.get(old_type)
            if new_type is None:
                continue

            if old_type in {
                "FaceLocalManualAdaptiveMaskCorrection",
                "SemanticObjectManualMaskCorrection",
            }:
                _migrate_manual_mask_apply(node, old_type, graph["links"])

            node["type"] = new_type
            properties = node.setdefault("properties", {})
            properties["Node name for S&R"] = new_type
            properties.pop("cnr_id", None)
            properties.pop("ver", None)
            changed.append((node["id"], old_type, new_type))

    if not changed:
        raise RuntimeError(f"no retired node IDs found in {source_path.name}")
    if _layout_signature(workflow) != original_layout:
        raise RuntimeError(f"layout changed while migrating {source_path.name}")
    if _semantic_link_signature(workflow) != original_links:
        raise RuntimeError(
            f"semantic links changed while migrating {source_path.name}"
        )

    retired_ids = set(LEGACY_TO_CANONICAL_NODE_IDS)
    residual = sorted(
        {
            node.get("type")
            for node in _walk_nodes(workflow)
            if node.get("type") in retired_ids
        }
    )
    if residual:
        raise RuntimeError(
            f"retired node IDs remain in {source_path.name}: {residual}"
        )

    _redact_serialized_secrets(workflow)
    output_path.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return changed


def migrate_workflow_set(source_dir: Path, output_dir: Path):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    missing = [
        source_dir / source_name
        for source_name in WORKFLOWS
        if not (source_dir / source_name).is_file()
    ]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"source directory is missing required workflows: {missing_names}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for source_name, output_name in WORKFLOWS.items():
        source_path = source_dir / source_name
        output_path = output_dir / output_name
        changes = migrate(source_path, output_path)
        report[output_name] = [
            {"node_id": node_id, "from": old_type, "to": new_type}
            for node_id, old_type, new_type in changes
        ]
    return report


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create pure-v2 RegionEdit copies of the three formal workflows."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="migrate three workflows from an explicitly selected directory",
    )
    migrate_parser.add_argument(
        "source_dir",
        type=Path,
        help="directory containing the three required legacy workflow files",
    )
    migrate_parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the pure-v2 copies (default: repository workflows)",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command != "migrate":
        raise RuntimeError(f"unsupported command: {args.command}")
    report = migrate_workflow_set(args.source_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
