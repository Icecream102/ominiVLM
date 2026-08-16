# 简历项目条目（多模态大模型核心算法岗：字节 / 阿里 / 腾讯）

> 数据截至 2026-08-16，全部由 AutoDL 单卡 RTX 4090（24GB）实测。
> 口径：COCO 官方 pycocoevalcap（val2017 全量 5000 张，含 METEOR）；VQAv2/OK-VQA 官方
> 3-of-10 规则；MMBench en/dev 全量 4329 题；POPE COCO 500 图 × 6 问。

## 版本一：算法 / 研究向（RLHF 对齐为主线，3-4 行）

**MiniMind-V：单卡可复现的多模态大模型训练与对齐全流程（65M → 3B → 7B）**

- 从零搭建 SigLIP2 + MLP Projection + LLM 的 VLM，跑通 Pretrain(1.27M) → SFT(2.9M) →
  多任务 SFT(231k) → GRPO/DPO → 官方 COCOEvalCap 评测闭环，单卡 24GB 全流程可复现；
- 诊断并修复 GRPO 奖励错位：代理奖励 +23.5% 但官方指标退化、KL 扩大 11 倍；通过自适应 KL +
  group 4→16 + CIDEr 对齐奖励，官方 BLEU-4 从 0.0067 修复至 0.0266（超过 SFT 0.0237），
  KL 压至 1/18；另实证 LLM-as-judge 奖励的 reward hacking（模型漂移到拒答腔）；
- 解决 VQA 短答案格式错位：0 → 专项 SFT 31.6% → 多任务 SFT 32.8%（同时官方 COCO CIDEr 0.64、
  BLEU-4 0.2271），并做数据配比消融（8.4% / 20.9% / 23.0%）定位 VQA-caption tradeoff；
- 完成规模跃迁：Qwen2.5-VL-3B LoRA（0.98% 参数、28 分钟）VQAv2 82.0%、MMBench 84.9%、
  COCO CIDEr 0.8364；7B QLoRA（单卡 1.5h）VQAv2 82.9%、MMBench 87.6%、CIDEr 0.9884、POPE 95.1%；
- 7B 对齐链：继续预训练（S4 混合语料使 OK-VQA +5.0pp、MMBench +0.19pp）→ 知识型 SFT
  （OK-VQA 48.6%）→ DPO 修复链（v5 塌缩 0.0% → v6 稳定 47.7%、COCO CIDEr 0.9973 全场最高），
  沉淀 DPO 六类故障诊断（数据偏置 / β / gather 越界 / reference 错配 / loss 误读 / 回答掩码）。

## 版本二：训练 / RLHF 对齐向（定向"核心算法岗"，3-4 行）

**多模态 RLHF 与对齐方法论（GRPO / DPO / 幻觉控制）**

- 自研 GRPO 实现（group-relative advantage、clipped objective、自适应 KL、KL safety stop），
  通过"奖励-评测对齐"修复链让官方 BLEU-4 首次超过 SFT（0.0266 > 0.0237），
  并量化 judge 奖励与官方指标相关性（Spearman 0.38–0.48）；
- 定位并修复 DPO 灾难性塌缩（MMBench 87.4% → 54.4%、退化 token），六项修复后 COCO CIDEr
  0.9973 全场最优、无退化 token，OK-VQA/MMBench 与 SFT 持平；
- 幻觉控制：POPE 全类型评测（positive/random/popular/adversarial），65M 数据配比-召回权衡
  曲线（v1/v2）+ 7B 幻觉 DPO（positive F1 0.949→0.957，随机负例 yes 率 1.9%→2.6%）；
- 偏好数据工程：judge 3B 采样 1.5k 对 → OK-VQA 9k 对 → 过滤平衡 11k 对（剔除 26.6% yes/no
  偏置），验证 β 与数据规模对 DPO 的敏感度（229 对负收益 → 9k 对转正）。

## 版本三：工程向（训练 / 评测基础设施，2-3 行）

**可复现多模态训练 / 评测流水线（单卡 24GB）**

- Pretrain → SFT → GRPO → DPO → 评测一键流水线：原子断点续训、JSON 配置校验、.done 标记、CI，
  中断精确续跑；Qwen2.5-VL NaViT 自定义 collator、bf16、梯度裁剪、cosine 调度、QLoRA；
- 接入官方 COCOEvalCap（val2017 全量，含 METEOR Java 兼容修复）、VQAv2 / OK-VQA（3-of-10）、
  MMBench 4329 题、POPE，统一脚本支持 65M / 3B / 7B 同口径对比 + black/shuffled 视觉依赖对照；
- 数据画像工具：语言分布 / 去重（minhash）/ 长度 / 图像规格，定位 14% 近似重复并给出方案；
- 全流程显存 / 吞吐统计：65M 49 分钟（9.6GB）、3B LoRA 28 分钟（17.9GB）、7B QLoRA 1.5h。

## 面试 30 秒叙事（RLHF 主线）

> 我独立搭了 65M 轻量 VLM 的完整训练闭环。关键发现是 GRPO 奖励错位：reward 上升 23.5%
> 但官方指标退化、KL 扩 11 倍。我通过 black/shuffled 对照确认视觉输入有效，再用自适应 KL、
> 更大 group 和 CIDEr 对齐奖励把官方 BLEU-4 从 0.0067 修到 0.0266（超过 SFT）。随后把同一套
> 方法论平移到 7B：LoRA/QLoRA 规模跃迁（MMBench 87.6%），并修复了 DPO 塌缩（v5 → v6，
> 无退化 token、COCO CIDEr 全场最高）。所有实验单卡 4090 可复现，官方口径评测。

## 必答追问

1. **为什么 65M？** 单卡预算内验证完整方法论闭环（对齐问题与大底座同构）；换 3B/7B 时
   训练流程可平移（LoRA/QLoRA），已在项目内完成规模跃迁验证。
2. **GRPO fix2 为什么 CIDEr 没超 SFT？** 奖励偏词面重合、缺语义多样性；下一步 judge 奖励 +
   held-out early stopping（judge-GRPO 已实验并记录 reward hacking 负结果）。
3. **VQA 0 分为什么不怀疑评测 bug？** 已验证加短答案提示词仍输出长句，属格式未对齐；
   口径与官方 3-of-10 一致；专项 SFT 后 0 → 31.6% 反证诊断正确。
4. **DPO 塌缩怎么定位的？** 先看数据（yes/no 26.6% 偏置）→ β 过弱梯度饱和 → 代码审查发现
   gather -100 索引 → reference 错配；逐项修复后 v6 稳定，每条都有消融/日志证据。
5. **METEOR 为什么跑不出来？** 新版 JVM 与 pycocoevalcap 1.2 meteor.py 不兼容，修复 wrapper
   （输入消毒 + 容错解析）后四指标全补齐。
6. **单卡 24GB 怎么训 7B？** NF4 QLoRA + 梯度检查点 + NaViT 动态分辨率，1.5h 完成 8k 多任务
   SFT；无多卡经验，如实标注。
7. **幻觉怎么控？** 65M 有配比-召回权衡曲线（1:3 过矫正、1:1 平衡）；7B 基座近饱和，
   幻觉 DPO 边际提升 positive F1；本质是视觉接地 + 规模问题。
8. **多任务 SFT 为什么能同时保住 caption 和 VQA？** 数据配方：任务配比消融 + 同图多问题/
   跨任务同图保留；单一任务 SFT 会灾难性遗忘（VQA 专项后 caption 归零）。

## 面向岗位的关键词命中

- 多模态对齐：Pretrain / SFT / GRPO / DPO / RLHF / reward hacking / 偏好数据；
- 训练：LoRA / QLoRA / 单卡 24GB / bf16 / NaViT collator / 断点续训；
- 评测：COCOEvalCap / VQAv2 / OK-VQA / MMBench / POPE / 幻觉 / 视觉依赖对照；
- 数据：配比消融 / 去重 / 画像 / 多任务混合；
- 研究素养：负结果如实呈现、诚实边界声明。

