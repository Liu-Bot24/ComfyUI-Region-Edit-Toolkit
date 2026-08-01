# 迁移到纯新版图像分区编辑节点

## 当前原则

- 插件只注册 42 个 `RegionEdit...` 节点。
- 43 个旧节点 ID 全部退役，不存在旧 ID 兼容注册。
- 旧 ID 与新 ID 的对应关系只用于迁移工作流，不参与 ComfyUI 运行。
- 迁移必须创建副本，不能覆盖用户的旧正式工作流。

## 已生成的三份副本

- [ComfyUI-Smart-Removal-RegionEdit-v2.json](../workflows/ComfyUI-Smart-Removal-RegionEdit-v2.json)
- [FaceLocal-Dual-Reference-Final-RegionEdit-v2.json](../workflows/FaceLocal-Dual-Reference-Final-RegionEdit-v2.json)
- [ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json](../workflows/ComfyUI-Semantic-Object-Replacement-RegionEdit-v2.json)

副本保留原节点 ID、节点位置、尺寸、分组、Subgraph 边界、链接 ID、链接的语义来源与去向，以及既有用户参数。迁移只替换退役节点类型及其搜索替换元数据；两个旧手涂应用节点合并到同一个新版节点时，脚本会补齐模式参数，并按输入名称同步调整受影响链接的目标槽位。

## 为什么是 43 个旧 ID 映射到 42 个新版 ID

新版按真实处理职责合并了两组重复实现：

- Smart 与 Semantic Object 的手涂遮罩画布统一为 `RegionEditMaskEditorCanvas`。
- Face 与 Semantic Object 的手涂结果应用统一为 `RegionEditMaskEditorApply`，通过明确的“强制核心锁”和“应用手涂”参数保留两种原行为。

其余只有名字相似、但端口或算法不等价的节点没有被强行合并。例如宽支持区域合成与普通严格坐标合成仍是两个节点。

完整对应关系见 [退役节点 ID 映射表](RETIRED_NODE_ID_MAP.md)。

## 安全迁移步骤

1. 关闭 ComfyUI。
2. 备份节点包、旧工作流和当前用户配置。
3. 安装当前纯新版节点包及依赖。
4. 重启 ComfyUI，确认 `/object_info` 中出现 42 个 `RegionEdit...` 节点，且没有由本包注册的旧 ID。
5. 加载迁移后的工作流副本，不要覆盖原工作流。
6. 保存并重新加载副本，检查 Subgraph、输入控件、预览、MaskEditor 和前端交互没有丢失。
7. 分别执行工作流中的最小关键路径，再执行完整路径。
8. 只有确认其他工作流不再依赖旧节点包后，才备份并禁用旧包。

## 自动迁移

仓库内迁移脚本从用户明确指定的目录读取旧工作流，并创建三个纯新版副本。源目录必须同时包含以下文件：

- `ComfyUI-Smart-Removal.json`
- `FaceLocal-Dual-Reference-Final.json`
- `ComfyUI-Semantic-Object-Replacement.json`

命令使用位置参数；第二个路径不填时，输出到仓库的 `workflows/`：

```powershell
python scripts/migrate_formal_workflows_to_pure_v2.py migrate "E:\旧工作流目录" "D:\新版副本输出目录"
```

脚本不读取仓库外的隐式恢复点。它会先确认三个源文件全部存在；缺少任意一个时不会写入任何副本。

脚本会拒绝以下情况：

- 源工作流没有可迁移的退役 ID；
- 迁移改变节点布局；
- 迁移改变链接所连接的语义端口；
- 迁移后仍残留退役节点 ID。

## 验证边界

仓库测试会检查：

- 注册表只有 42 个 `RegionEdit...` ID；
- 43 个退役 ID 都不在注册表；
- 映射目标全部存在；
- 非合并节点的序列化契约与旧实现一致；
- 三份副本的布局、节点数量、链接 ID 和语义连接保持不变；
- 每条链接记录与来源输出、目标输入两端的序列化引用完全闭合；
- 每个迁移节点的输入输出端口与新版类一致；
- 两类手涂工作流的原有开关语义得到保留。

这些检查属于源码、JSON 和轻量算法验证。插件安装、ComfyUI 重启、前端加载回存及真实图片执行必须在目标运行环境中另行完成，不能由静态通过代替。

## 回退

迁移副本出现问题时：

1. 停止使用副本；
2. 保留旧节点包；
3. 重新加载未修改的旧正式工作流；
4. 根据恢复点或 Git 提交恢复节点源码。

因为迁移不覆盖旧工作流，所以回退不需要从副本中反向转换。
