# Resume-ready project description

## 中文版本

**MiniMind-V 轻量级多模态大模型训练与对齐**

- 基于 SigLIP2 Vision Encoder、MLP Projection 与 65M MiniMind LLM 搭建端到端 VLM，完成 127 万图文样本 Pretrain、290 万混合指令样本 SFT 与 5000-step GRPO 后训练；支持 bfloat16、冻结策略、原子断点续训和一键可恢复流水线。
- 实现 COCO2017 统一评测框架，在固定 500 张验证图上统计 BLEU、METEOR、ROUGE-L、CIDEr-style、吞吐与显存，并通过正确图/全黑图/错配图对照量化视觉依赖，保留逐样本预测以支持误差分析。
- 针对 GRPO 中“代理奖励上升但下游 Caption 指标下降”的奖励错位问题，分析 KL 从前 500 步 0.076 增至末 500 步 0.838 的策略漂移，引入自适应 KL 系数、KL safety stop、可配置组合奖励和分项日志。
- 在单张 RTX 4090（24GB）上完成全量可复现实验：Pretrain 1.69h、SFT 6.28h、GRPO 0.50h；建立 JSON 实验配置、结果自动汇总、单元测试与 GitHub Actions CI，形成可审计的训练日志、模型卡和实验报告。

## 面试展开要点

1. 为什么冻结视觉编码器，以及 Projection-only Pretrain 的作用。
2. 为什么 SFT 只解冻 LLM 首尾层，如何平衡跨模态适配与语言能力遗忘。
3. GRPO 的 group-relative advantage、clipped objective、reference KL 如何实现。
4. 为什么 reward 上升不等于能力提升；如何用 held-out 指标、自适应 KL 和 early stopping 修复。
5. 为什么必须做 black/shuffled 对照，以及“输出发生变化”与“真正理解图像”的区别。

## 二、升级版（2026-08-12 下午）：多任务 VLM 与评测/数据/RLHF 补强

**一句话版**：在 65M 底座上通过 231k 多任务 SFT 拿到官方 COCO CIDEr 0.64
（+110×）、VQAv2 32.8%、MMBench 26.0%；补齐官方 METEOR 与 POPE 幻觉基准；
并用 Qwen3B judge 奖励做 GRPO v3，实证强模型奖励代理的迁移性。

### 简历条目（研究向 + 工程向）

- **多任务 VLM 训练**：构建 231k 多任务 SFT 数据（COCO caption + VQAv2 +
  OK-VQA + MMBench MCQ，minhash 画像去重），单阶段 SFT 让 65M 模型
  官方 COCO CIDEr 0.006 → 0.64、BLEU-4 0.2271、VQAv2 32.8%、MMBench 26.0%，
  同时解决“VQA 与 caption 互斥”的 tradeoff。
- **评测体系**：官方 COCOEvalCap 四指标全补齐（修复 METEOR 与新版 JVM 的
  兼容问题）、接入 OK-VQA 与 POPE 幻觉基准（65M 37.4% vs 3B 94.2%，
  量化小模型 yes 幻觉），全部 65M/3B 双口径。
- **RLHF 奖励升级**：实现 Qwen2.5-VL-3B judge 奖励版 GRPO（group-relative
  advantage + 自适应 KL + safety stop），分析 judge 与官方指标的 Spearman
  相关（r=0.38–0.48），800 步纯 judge 训练无 KL 漂移。

### 面试展开要点（新增）

6. 为什么多任务 SFT 能同时改善 caption 与 VQA——数据配方（任务配比、
   同图多问题保留、跨任务同图保留）如何影响 tradeoff。
7. 官方 METEOR 为什么跑不通（Java/parser 兼容）以及如何修复而不改口径。
8. POPE 的 yes 幻觉如何量化（yes-ratio / 负样本准确率），以及小模型
   幻觉更严重的可能机制（容量不足 → 先验偏向）。
9. judge 奖励为什么不直接用——奖励代理与评测指标的相关性分析、
   以及“优化代理不一定优化目标”的边界。
10. 用 3B LoRA 作为 judge/对比模型：0.98% 参数、28 分钟、VQAv2 82%、
    COCO CIDEr 0.836、POPE 94.2%。

## 三、第二轮升级版（2026-08-13）：7B QLoRA / DPO / 从零 VLM

**一句话版**：单卡 24GB 完成 7B QLoRA 多任务训练（官方 COCO CIDEr
0.988、MMBench 87.6%、VQAv2 82.9%、POPE 95.1%），judge 偏好 DPO 零损失，
并验证 freeze_llm=0 从零 VLM 预训练优于冻结投影版。

### 简历条目

- **7B QLoRA 多任务训练**：Qwen2.5-VL-7B + 4-bit NF4 QLoRA（1.13% 参数），
  8k 精选多任务数据（caption/VQA/幻觉/空间/OCR），单卡 24GB 1.5h 完成，
  四项基准全面超 3B LoRA（COCO CIDEr 0.836 → 0.988，MMBench 84.9% →
  87.6%，POPE 94.2% → 95.1%）。
- **从零 VLM 预训练**：随机 projector + 全 LLM 解冻（freeze_llm=0）
  在 1.27M 图文上对齐，COCO 四指标超越冻结投影路线（BLEU-4 0.2271 →
  0.2485），并用 58k 幻觉负样本把 POPE 从 37.4% 修到 78.5%
  （记录 yes→no 过矫正边界）。
- **偏好对齐**：3B judge 构建 1.5k 偏好对 + 手写 DPO（4-bit 参考模型），
  7B 能力零损失。

### 新增面试展开点

11. 单卡 24GB 训练 7B 的显存与时间预算（4-bit/梯度检查点/SDPA/分辨率
    权衡），以及 8k 样本 QLoRA 为何能逼近全参效果。
12. 幻觉负样本数据为何会过矫正（yes 77% → 7%）：先验 vs 召回 tradeoff、
    配比调优方向。
13. 为什么 VLM 对齐阶段解冻 LLM（freeze_llm=0）优于只训投影层——
    容量与任务复杂度关系。
14. DPO vs GRPO 的选择：离线偏好对为何更稳（避免在线采样被 judge
    hack），以及参考模型 4-bit 化的工程细节。
15. 评测升级：POPE 约束解码校准（ECE/AUROC）、text-only 鲁棒性、
    MMBench 按类拆分的分析方法论。

## 四、第三轮升级版（2026-08-14/15）：知识增强与 DPO 塌缩修复

**一句话版**：7B 继续预训练结论翻正（S4 混合语料 OK-VQA +5.0pp），知识型 SFT
OK-VQA 48.6% / MMBench 87.50%，并完整修复 DPO 灾难性塌缩（v5 退化 token 998/1000 →
v6 无退化、COCO CIDEr 0.9973 全场最高）。

### 简历条目

- **继续预训练-任务对齐度发现**：纯 caption 继续预训练（r=128、160k 样本）不迁移知识问答
  （OK-VQA 39.7%），混合任务语料 S4（caption+OK-VQA+multitask）在 OK-VQA +5.0pp、
  MMBench +0.19pp 双正——继续预训练增益取决于语料-任务对齐度；
- **DPO 六类故障诊断与修复**：数据偏置（26.6% yes/no）→ 过滤平衡 11k 对；β 0.02→0.1；
  gather -100 索引越界修复；reference 错配修复；loss 误读澄清；回答掩码修正。v6 稳定，
  COCO CIDEr 0.9973（全场最高）、OK-VQA 47.7%、MMBench 87.32%、退化 token 0/1000；
- **幻觉 DPO**：7B positive F1 0.949 → 0.957，随机负例 yes 率仅 2.6%，近饱和下的边际提升。

> 最新、最全的简历条目（面向字节/阿里/腾讯多模态核心算法岗）见
> [RESUME_ENTRIES.md](RESUME_ENTRIES.md)，完整结果表见
> [PROJECT_RESULTS_SUMMARY.md](PROJECT_RESULTS_SUMMARY.md)。

## 使用边界

简历中应明确这是 65M 轻量模型的单卡研究与工程实验，不应写成“大规模工业 VLM 训练平台”。优势在于完整闭环、可复现性、算法诊断和诚实的负实验分析。
