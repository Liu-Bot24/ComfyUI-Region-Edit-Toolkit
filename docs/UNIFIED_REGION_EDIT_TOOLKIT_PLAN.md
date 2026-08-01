# 图像分区编辑工具包：纯 v2 实施方案

## 目标

把 Smart Removal、Face Local Edit Transfer 和 Semantic Object Replacement 中的自研节点收进一个安装包，并按节点真实处理职责重新命名和组织。

本次实施只做三件事：

1. 保留三套既有工作流实际依赖的能力；
2. 合并真正重复的节点；
3. 创建不覆盖原件的纯新版工作流副本。

不新增与三套现有工作流无关的功能。

## 最终注册边界

插件只注册 42 个 `RegionEdit...` 节点。43 个旧 ID 全部退役，只在迁移表和迁移测试中出现。

注册层位于：

- `region_edit_toolkit/canonical_nodes.py`
- `region_edit_toolkit/registry.py`

旧实现类仍可作为内部算法来源被纯新版外壳继承，但内部类名不会进入 ComfyUI 注册表。

## 为什么不是原先估计的 29 个节点

逐项比对正式工作流、序列化端口和算法后，发现原先 29 个规范节点缺少 13 项不可丢失的真实能力：

- 自动／手工／保护遮罩的多模式合成；
- 旧式基础批量裁块；
- 分块参数档位解析；
- 保留约束 Prompt 构建；
- 人脸自适应差异与核心；
- 人脸阈值差异与核心；
- 外边界羽化与强制核心保护；
- 宽支持区域严格合成；
- 惰性参考图来源选择；
- Prompt 路线选择；
- 人脸编辑 Prompt 合约；
- 人像 Prompt 质量检查；
- 替换编辑 Prompt 合约。

这些能力不能靠改名或错误接线替代，因此补成正式纯新版节点。最终 42 个节点是功能审计结果，不是为了凑数量。

## 真正合并的重复节点

### 手涂遮罩画布

Smart 与 Semantic Object 的旧画布具有相同本体功能，统一为：

`RegionEditMaskEditorCanvas`

### 应用手涂遮罩

Face 与 Semantic Object 的旧应用节点共享同一处理骨架，统一为：

`RegionEditMaskEditorApply`

新版显式区分：

- 是否锁定强制核心；
- 是否应用当前手涂结果。

这样一个节点可以保留两条工作流原行为，又不会把场景名称写进节点职责。

## 保持分开的相近能力

以下节点名字相近但契约或算法不同，不能强行合并：

- `RegionEditMaskSetCompose` 与 `RegionEditInteractiveMaskCompose`
- `RegionEditAlignedDifferenceMask` 与两个人脸差异节点
- `RegionEditBoundaryFeather4Side` 与 `RegionEditProtectedOuterBoundaryFeather`
- `RegionEditStrictCoordinateComposite` 与 `RegionEditWideSupportStrictComposite`
- `RegionEditBasicTileBatchPrepare` 与 `RegionEditTileBatchPrepare`

保留它们能避免端口错位、核心保护丢失、Subgraph 断线或工作流语义变化。

## 工作流迁移规则

三份旧正式工作流先进入只读恢复点，再生成三个纯新版副本：

- `workflows/ComfyUI-Smart-Removal-RegionEdit-v2.json`
- `workflows/FaceLocal-Dual-Reference-Final-RegionEdit-v2.json`
- `workflows/ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json`

迁移保留：

- 节点 ID；
- 节点位置、大小、折叠状态和分组；
- Subgraph 输入输出边界；
- 所有 link ID 与语义连接；
- 原有 widget 值和业务 Prompt。

迁移只改变：

- 退役节点的 `type`；
- 对应的 `Node name for S&R`；
- 两类手涂应用节点为保留原语义所必需的显式模式字段。
- 合并节点输入顺序变化时，与输入名称对应的 `target_slot`。

## 静态验收

必须满足：

- 注册表恰好包含 42 个 `RegionEdit...` ID；
- 43 个退役 ID 与注册表完全不相交；
- 映射表的每个目标都已注册；
- 新版节点的输入输出契约与迁移后的 JSON 一致；
- 三份副本没有布局、链接 ID 或语义连接变化；
- 每条链接与节点输入输出两端的引用完全闭合；
- 迁移脚本可重复执行并得到相同结果；
- Python 编译、JavaScript 语法和全量测试通过。

## 运行态验收

安装当前源码并重启 ComfyUI 后，再完成：

1. `/object_info` 注册核验；
2. 三份副本的加载、保存、重载；
3. 参考图来源选择和 MaskEditor 前端交互；
4. 三条工作流各自的最小关键路径；
5. 完整工作流执行。

静态验收不能冒充运行态验收。
