# SGCSR local_topk 消融实验

## 实验目的

验证 SGCSR local attention 的 top-k 空间约束是否过强，并比较不同候选 token 上限对模型效果的影响。

## 变量设置

固定：

- `local_radius=0.15`
- 其他模型结构、训练目标、数据划分和训练配置保持一致

改变：

- `local_topk=8`
- `local_topk=16`
- `local_topk=32`
- `local_topk=64`
- `local_topk=0`

其中，`local_topk=0` 表示关闭 top-k 限制；如果同时保留 `local_radius=0.15`，attention 仍然只读取半径范围内的 token。

## 参数传递链

训练时：

```text
--local_topk -> train_sgcsr.py / train_sgcsr_pope_adapt.py
             -> SourceGuidedCompactSemanticReconstructor
             -> ReconstructionBlock
             -> SpatialCrossAttention
```

评估时，`evaluate_sgcsr_pretrain.py` 和 `evaluate_sgcsr_pope.py` 支持通过命令行覆盖 checkpoint 中保存的 `local_topk`：

```bash
--local_topk 8
--local_topk 16
--local_topk 32
--local_topk 64
--local_topk 0
```

评估命令不传该参数时，继续使用 checkpoint 中保存的值，保证旧 checkpoint 的默认评估行为不变。评估输出 JSON 会记录实际生效的 `local_topk` 和 `local_radius`。

## 运行说明

`run_local_topk_ablation.sh` 只提供命令模板，不会自动执行。运行前请替换其中的模型路径、数据路径、初始化 checkpoint 和输出目录。

如果从已有 SGCSR checkpoint 初始化训练，并且该 checkpoint 的 `local_topk` 与当前消融值不同，需要保留命令中的 `--allow_checkpoint_config_mismatch`。该选项只允许有意进行 locality ablation，不会改变 checkpoint 的权重格式。

每组实验应使用独立的 `output_dir`，并在结果文件中记录对应的 `local_topk`。

## 约束

本消融不修改：

- `SourceAwareTokenEncoder`
- learnable query tokens
- `SpatialCrossAttention` 的计算逻辑
- `SpatialSmoothBlock`
- `FeedForward`
- LayerNorm
- feature distillation、task loss、logit distillation 及其权重
