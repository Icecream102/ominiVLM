# Resume-ready project description

## 中文版本

**MiniMind-V 轻量级多模态大模型训练与对齐**

- 基于 SigLIP2 Vision Encoder、MLP Projection 与 65M MiniMind LLM 搭建端到端 VLM，完成 127 万图文样本 Pretrain、290 万混合指令样本 SFT 与 5000-step GRPO 后训练；支持 bfloat16、冻结策略、原子断点续训和一键可恢复流水线。
- 实现 COCO2017 统一评测框架，在固定 500 张验证图上统计 BLEU、METEOR、ROUGE-L、CIDEr-style、吞吐与显存，并通过正确图/全黑图/错配图对照量化视觉依赖，保留逐样本预测以支持误差分析。
- 针对 GRPO 中“代理奖励上升但下游 Caption 指标下降”的奖励错位问题，分析 KL 从前 500 步 0.076 增至末 500 步 0.838 的策略漂移，引入自适应 KL 系数、KL safety stop、可配置组合奖励和分项日志。
- 在单张 RTX 5090 上完成全量可复现实验：Pretrain 1.69h、SFT 6.28h、GRPO 0.50h；建立 JSON 实验配置、结果自动汇总、单元测试与 GitHub Actions CI，形成可审计的训练日志、模型卡和实验报告。

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

## 使用边界

简历中应明确这是 65M 轻量模型的单卡研究与工程实验，不应写成“大规模工业 VLM 训练平台”。优势在于完整闭环、可复现性、算法诊断和诚实的负实验分析。
