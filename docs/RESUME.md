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

## 使用边界

简历中应明确这是 65M 轻量模型的单卡研究与工程实验，不应写成“大规模工业 VLM 训练平台”。优势在于完整闭环、可复现性、算法诊断和诚实的负实验分析。
