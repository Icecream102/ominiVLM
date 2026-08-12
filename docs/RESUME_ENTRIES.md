# 简历项目条目（多模态大模型算法岗）

> 数据截至 2026-08-12，含官方 COCOEvalCap 评测与 VQA 专项 SFT 计划。
> 口径说明：所有指标为 65M 轻量底座在单卡 24GB 上的受控对比；官方 METEOR 因运行时不可用，使用内部 METEOR-exact 并标注。

## 版本一：算法/研究向（约 3 行）

**MiniMind-V 轻量多模态大模型训练与对齐（65M）**

- 独立完成 SigLIP2+MLP Projection+LLM 架构的 Pretrain(1.27M)→SFT(2.9M)→GRPO(5000 步)→官方 COCOEvalCap 评测全流程，单卡 24GB 可复现；
- 诊断并修复 GRPO 奖励错位：代理奖励上升 23.5% 但 CIDEr 退化、KL 扩大 11×；通过自适应 KL、group 4→16、CIDEr 对齐奖励将官方 BLEU-4 从 0.0067 修复到 0.0266（超过 SFT 的 0.0237），KL 压至 1/18；
- 完成 VQAv2 评测管线与多任务对齐闭环：基线 0 → VQA 专项 SFT 31.6%（暴露 caption 灾难性遗忘）→ 混合 SFT 20.9% 且官方 COCO CIDEr 0.0067 全场最优（双任务保留）；COCO 官方 BLEU/ROUGE/CIDEr + 视觉依赖 black/shuffled 对照。
- **规模实验**：Qwen2.5-VL-3B + LoRA（r=16，37.2M 可训练参数/0.98%，625 步/28 分钟/单卡 24GB）VQAv2 82.0%、官方 COCO CIDEr 0.8364（较 65M 提升 120×）；完成数据配比消融（1:3/1:1/3:1）与数据画像（去重率 86%、中英均衡）。
- **综合基准**：MMBench en/dev 4329 题全量 84.9%（3B LoRA）vs 65M 12.0%，统一评测脚本支持两套模型同口径对比。
- **LoRA 消融**：3B 零样本 83.95% vs LoRA 84.9%（+0.94pp）——量化任务适配与基座能力的边界，VQA 微调主要提升目标任务。

## 版本二：工程向（约 3 行）

**可复现多模态训练/评测流水线**

- 搭建 Pretrain→SFT→GRPO→评测一键流水线：原子断点续训、JSON 配置校验、.done 标记、CI，中断后可精确续跑；
- 接入官方 COCOEvalCap（pycocoevalcap，val2017 全量 5000 张）与 VQAv2 评测（图片直读公共数据盘 zip，零本地磁盘占用），支持 METEOR 超时保护；
- 搭建 Qwen2.5-VL LoRA 训练管线（NaViT 自定义 collator、bf16、梯度裁剪、cosine 调度）与数据画像/配比工具；
- 全流程含吞吐/显存统计、black/shuffled 视觉依赖对照、逐样本预测与可视化，单卡 RTX 4090 端到端可复现。

## 训练岗定向要点（多模态训练/对齐方向）

- 训练配方有实验支撑：数据配比消融、冻结策略（首尾层 vs LoRA）、自适应 KL、reward 权重设计；
- 规模效率数据：3B LoRA 0.98% 参数在 24GB 单卡 28 分钟收敛，VQA 82.0%、COCO CIDEr 0.8364；
- 训练工程细节：Qwen2.5-VL NaViT collator 维度处理、HF Dataset 张量转换、CUDA 设备断言排查；
- 数据工程：语言分布/去重/长度画像 + 配比消融，发现 ~14% 重复并给出 minhash 方案；
- 明确边界：单卡 4090，无多卡 FSDP/DeepSpeed 经验（如实标注，不强称）。

## 面试 30 秒叙事

> 我独立搭建了 65M 轻量 VLM 的完整训练闭环。关键发现是 GRPO 奖励错位：reward 上升但指标退化、KL 扩大 11 倍。我通过对照实验确认模型确实使用视觉输入，再用自适应 KL、更大 group 和 CIDEr 对齐奖励把官方 BLEU-4 从 0.0067 修复到 0.0266（超过 SFT），并搭好 VQA 评测定位到短答案格式对齐问题。

## 必答追问（提前准备）

1. 为什么 65M？→ 在单卡预算内验证完整方法论闭环；换大底座时训练流程可平移（LoRA/多卡）。
2. fix2 为什么 CIDEr 没超 SFT？→ 奖励偏词面重合、缺语义多样性；下一步 judge 模型奖励 + held-out early stopping。
3. VQA 0 分为什么不怀疑评测 bug？→ 已验证：加短答案提示词仍输出长句，属模型行为；口径与官方 3-of-10 一致。
4. METEOR 为什么没跑出来？→ Java/Stanford parser 兼容问题，已加超时保护，用内部 METEOR-exact 并标注。
5. 如果只做一个实验？→ VQA 专项 SFT：把已诊断的负结果变成第二个修复闭环。
