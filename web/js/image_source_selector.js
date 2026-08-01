import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPE = "RegionEditReferenceSourceSelector";
const STAGE_ZERO_KEY = "face-crop-source-selection-v3";
const SUBGRAPH_INPUT_NODE_ID = -10;
const BYPASS_MODE = 4;
const NEVER_MODE = 2;
const lastSignatures = new Map();

function rootGraph() {
  return app.rootGraph ?? app.graph;
}

function graphLink(graph, linkId) {
  if (linkId == null) return null;
  return (
    graph?.getLink?.(linkId) ??
    graph?._links?.get?.(linkId) ??
    graph?.links?.[linkId] ??
    null
  );
}

function linkedNode(node, inputIndex) {
  const input = node?.inputs?.[inputIndex];
  const graph = node?.graph ?? rootGraph();
  const link = graphLink(graph, input?.link);
  if (!link) return null;
  return graph?.getNodeById?.(link.origin_id) ?? null;
}

function booleanInputValue(node, inputIndex) {
  const source = linkedNode(node, inputIndex);
  if (!source) return false;
  const widget = source.widgets?.find((item) => item.name === "value") ?? source.widgets?.[0];
  return Boolean(widget?.value);
}

function selectedSource(node) {
  if (node.mode === NEVER_MODE) return null;
  if (node.mode === BYPASS_MODE) return linkedNode(node, 0);
  const useGptReference = booleanInputValue(node, 2) || booleanInputValue(node, 3);
  return linkedNode(node, useGptReference ? 1 : 0);
}

function selectorInHost(host) {
  if (host?.properties?.face_local_subgraph_key !== STAGE_ZERO_KEY) return null;
  const nodes = host.subgraph?._nodes ?? host.subgraph?.nodes ?? [];
  return nodes.find((node) => node.type === NODE_TYPE) ?? null;
}

function hostSourceForSelectorInput(host, selector, selectorInputIndex) {
  const input = selector?.inputs?.[selectorInputIndex];
  const innerLink = graphLink(selector?.graph, input?.link);
  if (!innerLink || Number(innerLink.origin_id) !== SUBGRAPH_INPUT_NODE_ID) return null;
  return host.getInputNode?.(innerLink.origin_slot) ?? null;
}

function booleanSelectorInputValue(host, selector, selectorInputIndex) {
  const source = hostSourceForSelectorInput(host, selector, selectorInputIndex);
  const widget = source?.widgets?.find((item) => item.name === "value") ?? source?.widgets?.[0];
  return Boolean(widget?.value);
}

function selectedNestedSource(host, selector) {
  if (host.mode === NEVER_MODE || selector.mode === NEVER_MODE) return null;
  if (host.mode === BYPASS_MODE || selector.mode === BYPASS_MODE) {
    return hostSourceForSelectorInput(host, selector, 0);
  }
  const useGptReference =
    booleanSelectorInputValue(host, selector, 2) ||
    booleanSelectorInputValue(host, selector, 3);
  return hostSourceForSelectorInput(host, selector, useGptReference ? 1 : 0);
}

function locatorForNode(node) {
  const graph = node?.graph;
  return graph && graph !== rootGraph() ? `${graph.id}:${node.id}` : String(node.id);
}

function loadImageFallbackOutput(source) {
  const imageWidget = source?.widgets?.find((item) => item.name === "image") ?? source?.widgets?.[0];
  const filename = imageWidget?.value;
  if (typeof filename !== "string" || !filename.trim()) return null;
  return {
    images: [{ filename, subfolder: "", type: "input" }],
  };
}

function sourceOutput(source) {
  if (!source) return null;
  return app.nodeOutputs?.[locatorForNode(source)] ?? loadImageFallbackOutput(source);
}

function sourceFilename(source) {
  return (
    source?.widgets?.find((item) => item.name === "image")?.value ??
    source?.widgets?.[0]?.value ??
    ""
  );
}

function dispatchOutput(executionId, locatorId, output) {
  if (!output) {
    if (app.nodeOutputs) delete app.nodeOutputs[locatorId];
  } else {
    app.nodeOutputs ??= {};
    app.nodeOutputs[locatorId] = output;
  }
  api.dispatchEvent(
    new CustomEvent("executed", {
      detail: { node: executionId, output: output ?? { images: [] } },
    }),
  );
  rootGraph()?.setDirtyCanvas(true, true);
}

function syncSelector(node) {
  if (!node || node.type !== NODE_TYPE) return;
  const source = selectedSource(node);
  const output = sourceOutput(source);
  const signature = JSON.stringify({
    mode: node.mode,
    sourceId: source?.id ?? null,
    sourceFilename: sourceFilename(source),
    images: output?.images ?? null,
  });
  const locatorId = locatorForNode(node);
  if (lastSignatures.get(locatorId) === signature) return;
  lastSignatures.set(locatorId, signature);
  node.imgs = source?.imgs ?? null;
  dispatchOutput(String(node.id), locatorId, output);
}

function syncNestedSelector(host, selector) {
  const source = selectedNestedSource(host, selector);
  const output = sourceOutput(source);
  const locatorId = `${host.subgraph.id}:${selector.id}`;
  const executionId = `${host.id}:${selector.id}`;
  const signature = JSON.stringify({
    hostMode: host.mode,
    selectorMode: selector.mode,
    sourceId: source?.id ?? null,
    sourceFilename: sourceFilename(source),
    images: output?.images ?? null,
  });
  if (lastSignatures.get(executionId) === signature) return;
  lastSignatures.set(executionId, signature);
  selector.imgs = source?.imgs ?? null;
  dispatchOutput(executionId, locatorId, output);
}

function syncEasyUiMirrors() {
  const graph = rootGraph();
  for (const mirrorNode of graphNodes(graph)) {
    if (mirrorNode.type !== "UINode") continue;
    const originalId = mirrorNode.properties?.original_node_id;
    if (originalId == null) continue;
    const originalNode = graph.getNodeById?.(originalId);
    if (!originalNode?.widgets?.length || !mirrorNode.widgets?.length) continue;

    const hidden = new Set(mirrorNode.properties?.hiddenWidgets ?? []);
    const originalWidgets = originalNode.widgets.filter(
      (widget) => !hidden.has(widget.name) && !widget.computedDisabled,
    );
    for (let index = 0; index < originalWidgets.length; index += 1) {
      const originalWidget = originalWidgets[index];
      const mirrorWidget =
        mirrorNode.widgets.find((widget) => widget.name === originalWidget.name) ??
        mirrorNode.widgets[index];
      if (!mirrorWidget) continue;

      if (mirrorWidget._regionEditMirrorSource !== originalWidget) {
        const previousCallback = mirrorWidget.callback;
        mirrorWidget._regionEditMirrorSource = originalWidget;
        mirrorWidget.callback = function (value, ...args) {
          if (mirrorWidget._regionEditMirrorSyncing) return;
          originalWidget.value = value;
          if (typeof previousCallback === "function") {
            previousCallback.call(this, value, ...args);
          } else if (typeof originalWidget.callback === "function") {
            originalWidget.callback.call(originalWidget, value, ...args);
          }
          graph.setDirtyCanvas?.(true, true);
        };
      }

      if (!Object.is(mirrorWidget.value, originalWidget.value)) {
        mirrorWidget._regionEditMirrorSyncing = true;
        mirrorWidget.value = originalWidget.value;
        mirrorWidget._regionEditMirrorSyncing = false;
      }
    }
  }
}

app.registerExtension({
  name: "RegionEditToolkit.ReferenceSourceSelectorPreview",
  nodeCreated(node) {
    if (node.type === NODE_TYPE) syncSelector(node);
  },
  setup() {
    window.setInterval(() => {
      for (const node of graphNodes(rootGraph())) {
        if (node.type === NODE_TYPE) syncSelector(node);
        const selector = selectorInHost(node);
        if (selector) syncNestedSelector(node, selector);
      }
      syncEasyUiMirrors();
    }, 150);
  },
});

const SIMPLE_GROUP_CONTROLLER_TYPE = "RegionEditSimpleGroupBypassController";
const ACTIVE_MODE = 0;
const BYPASS_MODE_FOR_GROUP = 4;
const ADD_GROUP_PLACEHOLDER = "＋ 添加区域…";
const REMOVE_GROUP_PLACEHOLDER = "－ 删除区域…";

function graphGroups(graph) {
  const groups = graph?._groups ?? graph?.groups ?? [];
  if (Array.isArray(groups)) return groups;
  if (typeof groups.values === "function") return Array.from(groups.values());
  return Array.from(groups, (entry) =>
    Array.isArray(entry) && entry.length === 2 ? entry[1] : entry,
  );
}

function graphNodes(graph) {
  const nodes = graph?._nodes ?? graph?.nodes ?? [];
  if (Array.isArray(nodes)) return nodes;
  if (typeof nodes.values === "function") return Array.from(nodes.values());
  return Array.from(nodes, (entry) =>
    Array.isArray(entry) && entry.length === 2 ? entry[1] : entry,
  );
}

function setNodeModeDeep(node, mode) {
  const pending = [node];
  const seen = new Set();
  while (pending.length) {
    const current = pending.pop();
    if (!current || seen.has(current)) continue;
    seen.add(current);
    if ("mode" in current) current.mode = mode;
    const subgraph = current.subgraph ?? current._subgraph;
    if (subgraph) pending.push(...graphNodes(subgraph));
  }
}

function applyGroupMode(graph, target, enabled) {
  const mode = enabled ? ACTIVE_MODE : BYPASS_MODE_FOR_GROUP;
  for (const group of graphGroups(graph)) {
    if (group.title !== target.title) continue;
    group.recomputeInsideNodes?.();
    const members = group._children
      ? Array.from(group._children)
      : group._nodes ?? group.nodes ?? [];
    for (const node of members) {
      if (node && "mode" in node) setNodeModeDeep(node, mode);
    }
  }
  graph?.setDirtyCanvas?.(true, true);
}

class RegionEditSimpleGroupBypassController extends LGraphNode {
  constructor(title = "分组控制") {
    super(title);
    this.isVirtualNode = true;
    this.serialize_widgets = false;
    this.properties ??= {};
    this.properties.targets ??= [];
    this.size = [420, 100];
    this.buildWidgets();
  }

  normalizeTargets() {
    if (!Array.isArray(this.properties.targets)) this.properties.targets = [];
    this.properties.targets = this.properties.targets
      .filter((target) => typeof target?.title === "string" && target.title)
      .map((target) => ({
        title: target.title,
        label: target.label || target.title,
        enabled: Boolean(target.enabled),
      }));
  }

  buildWidgets() {
    this.normalizeTargets();
    this.widgets = [];
    for (const target of this.properties.targets) {
      this.addWidget(
        "toggle",
        target.label,
        target.enabled,
        (enabled) => {
          target.enabled = Boolean(enabled);
          applyGroupMode(this.graph ?? rootGraph(), target, target.enabled);
        },
        { on: "运行", off: "忽略" },
      );
    }
    this.addWidget(
      "combo",
      "添加区域",
      ADD_GROUP_PLACEHOLDER,
      (title) => {
        if (
          title &&
          title !== ADD_GROUP_PLACEHOLDER &&
          !this.properties.targets.some((target) => target.title === title)
        ) {
          this.properties.targets.push({ title, label: title, enabled: true });
          this.buildWidgets();
          const target = this.properties.targets.find((item) => item.title === title);
          if (target) applyGroupMode(this.graph ?? rootGraph(), target, true);
        }
      },
      {
        values: () => [
          ADD_GROUP_PLACEHOLDER,
          ...graphGroups(this.graph ?? rootGraph())
            .map((group) => group.title)
            .filter(
              (title) =>
                title &&
                !this.properties.targets.some((target) => target.title === title),
            ),
        ],
      },
    );
    if (this.properties.targets.length) {
      this.addWidget(
        "combo",
        "删除区域",
        REMOVE_GROUP_PLACEHOLDER,
        (title) => {
          if (title && title !== REMOVE_GROUP_PLACEHOLDER) {
            this.properties.targets = this.properties.targets.filter(
              (target) => target.title !== title,
            );
            this.buildWidgets();
          }
        },
        {
          values: () => [
            REMOVE_GROUP_PLACEHOLDER,
            ...this.properties.targets.map((target) => target.title),
          ],
        },
      );
    }
    this.setDirtyCanvas?.(true, true);
  }

  applyAll() {
    const graph = this.graph ?? rootGraph();
    for (const target of this.properties.targets) {
      applyGroupMode(graph, target, target.enabled);
    }
  }

  applyToGraph() {
    this.applyAll();
  }

  onAdded() {
    window.setTimeout(() => this.applyAll(), 0);
  }

  onConfigure() {
    this.buildWidgets();
    window.setTimeout(() => this.applyAll(), 0);
  }
}

RegionEditSimpleGroupBypassController.title = "分组控制（简洁）";
RegionEditSimpleGroupBypassController.category = "RegionEdit";
RegionEditSimpleGroupBypassController.collapsable = true;

app.registerExtension({
  name: "RegionEditToolkit.SimpleGroupBypassController",
  registerCustomNodes() {
    LiteGraph.registerNodeType(
      SIMPLE_GROUP_CONTROLLER_TYPE,
      RegionEditSimpleGroupBypassController,
    );
  },
});
