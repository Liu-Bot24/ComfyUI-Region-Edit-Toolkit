from __future__ import annotations

from .canonical_nodes import (
    CANONICAL_NODE_CLASS_MAPPINGS,
    CANONICAL_NODE_DISPLAY_NAME_MAPPINGS,
)


def _merge_unique(*mappings):
    merged = {}
    for mapping in mappings:
        overlap = set(merged).intersection(mapping)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise RuntimeError(f"duplicate ComfyUI node class IDs: {names}")
        merged.update(mapping)
    return merged


NODE_CLASS_MAPPINGS = _merge_unique(
    CANONICAL_NODE_CLASS_MAPPINGS,
)

NODE_DISPLAY_NAME_MAPPINGS = _merge_unique(
    CANONICAL_NODE_DISPLAY_NAME_MAPPINGS,
)

if set(NODE_CLASS_MAPPINGS) != set(NODE_DISPLAY_NAME_MAPPINGS):
    missing_display = sorted(
        set(NODE_CLASS_MAPPINGS).difference(NODE_DISPLAY_NAME_MAPPINGS)
    )
    orphan_display = sorted(
        set(NODE_DISPLAY_NAME_MAPPINGS).difference(NODE_CLASS_MAPPINGS)
    )
    raise RuntimeError(
        "node/display registry mismatch: "
        f"missing_display={missing_display}, orphan_display={orphan_display}"
    )

if len(CANONICAL_NODE_CLASS_MAPPINGS) != 42:
    raise RuntimeError("pure v2 registry must contain exactly 42 canonical class IDs")
