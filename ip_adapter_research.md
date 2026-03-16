# Wan2.1 参考图像条件注入：技术调研与后续计划

> 目标：在 Wan2.1 T2V 模型的 Cross-Attention 层实现 IP-Adapter 风格的参考图像注入，用于 PVTT（产品视频模板迁移）任务。

---

## 1. 方案选型与调研结论

### 1.1 原参考方案回顾

项目初期参考的方案均基于 UNet 架构，与 Wan2.1 的 DiT 架构存在较大差异：

| 方案 | 架构基础 | 核心思路 |
|------|---------|---------|
| IP-Adapter (Ye et al., 2308.06721) | SD1.5/SDXL (UNet) | 解耦 Cross-Attention，新增图像 K/V 投影 |
| InstantID (2401.07519) | SDXL (UNet) | IdentityNet + IP-Adapter，人脸身份保持 |
| ConsistI2V (2402.04324) | UNet-based VDM | 首帧条件注入 + 时序窗口注意力 |
| ID-Animator (2404.15275) | UNet-based VDM | 面部适配器 + 文本-图像融合 |

### 1.2 Wan2.1/2.2 生态中的现有方案

调研发现 DiT 生态已发展出原生的参考图像注入方案：

| 方案 | 来源 | 基础模型 | 架构方式 | 开源状态 |
|------|------|---------|---------|---------|
| **VACE R2V** | 阿里官方, ICCV 2025 | Wan2.1 | Context Adapter 旁路注入，冻结基础模型 | 代码+权重 |
| **Saber** | arXiv 2512.06905 | Wan2.1-14B | Masked Training + 时序 latent 拼接，零样本 R2V | 代码 |
| **Phantom-Wan** | ByteDance, ICCV 2025 | Wan2.1 | MMDiT 结构，图像/视频/文本分支 | Wan2.1 权重 |
| **Lynx** | ByteDance, CVPR 2026 | Wan2.1-14B | 双适配器：ArcFace ID + VAE Ref | 代码+权重 |
| **IPAdapterWAN** | 社区 | Wan2.1 | SigLIP2 + InstantX SD3.5 权重适配 | ComfyUI 插件 |
| **Wan2.2 I2V** | 阿里官方 | Wan2.2 | CLIP 视觉编码器 + VAE 首帧条件化 | 官方模型 |
| **HunyuanCustom** | 腾讯 | HunyuanVideo-13B | LLaVA 文本-图像融合 + 时序拼接 | 代码+权重 |

### 1.3 VACE 方案实验结论（组内同学已验证）

VACE 1.3B 在 PVTT 场景下效果不足：

- **成功率极低**：23 个采样任务仅 3 个（13%）正常
- **主体一致性不足**：生成的产品外观与参考图差异大，Canny 图嵌入仅轻微提升
- **两大失败模式**：
  - 杂乱噪声/闪烁（47.8%）：mask 帧数不足、bbox 覆盖率过大、bbox 帧间剧变
  - 画面滚动（39.1%）：bbox 大幅垂直位移
- **根本原因**：VACE 的"擦除+重绘"机制依赖 mask 和 bbox，大面积 mask 导致上下文缺失，物体与环境纠缠

### 1.4 为什么选 T2V + IP-Adapter 而非 I2V + Text

| | I2V（图生视频） | T2V + IP-Adapter（文本生成 + 图像注入） |
|---|---|---|
| 图像作用 | 作为**第一帧**，信息沿时间衰减 | 作为**全局语义条件**，每帧都受约束 |
| 文本控制 | 弱，被图像特征压制 | 强，文本和图像解耦控制 |
| 主体一致性 | 后半段外观漂移 | 全程保持 |
| PVTT 适用性 | 差——需在已有视频轨迹上替换物体 | 好——全局外观引导 + 独立动作控制 |

---

## 2. IP-Adapter 技术方案

### 2.1 架构设计

在 Wan2.1 T2V 的 30 层 DiT Block 的 Cross-Attention 中新增图像注入分支：

```python
# 原始文本分支（保持不变）
Q     = W_q @ hidden_states          # hidden_states: 视频特征, dim=1536
K_txt = W_k @ text_emb               # text_emb: T5 编码器输出
V_txt = W_v @ text_emb
Attn_text = Softmax(Q @ K_txt^T) @ V_txt

# 新增图像分支（IP-Adapter）
K_img = W_k_img @ image_tokens       # 新增参数, image_tokens 来自 Resampler
V_img = W_v_img @ image_tokens       # 新增参数
Attn_img = Softmax(Q @ K_img^T) @ V_img

Out = Attn_text + lambda * Attn_img   # lambda 可学习或固定
```

关键组件：

| 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 图像编码器 | 参考图像 (224x224) | patch tokens (~257 个) | CLIP ViT-H/14 或 DINOv2 ViT-L/14，冻结 |
| Perceiver Resampler | 257 patch tokens | 16~64 个 token (dim=1536) | 压缩图像 token 到固定长度，对齐 DiT hidden_dim |
| W_k_img / W_v_img | image_tokens (dim=1536) | K/V (dim=1536) | 每层 Cross-Attention 新增两个投影矩阵 |

### 2.2 已确认的架构参数（来自 LoRA 微调实验）

| 参数 | 值 |
|------|-----|
| DiT Block 数量 | 30（blocks.0 ~ blocks.29） |
| Cross-Attention hidden_dim | 1536 |
| FFN 中间维度 | 8960 |
| Self-Attention / Cross-Attention | 同维度，独立投影 |
| 训练范式 | Rectified Flow（FlowMatchSFTLoss） |

### 2.3 训练策略

- 冻结：基础 DiT + 图像编码器
- 训练：Perceiver Resampler + 30 层 W_k_img / W_v_img + 可选 LoRA
- CFG：20% 概率丢弃图像条件（输入零向量），支持 Classifier-Free Guidance
- 损失函数：FlowMatchSFTLoss（与 Wan2.1 训练范式一致）

### 2.4 参考实现

| 项目 | 参考价值 |
|------|---------|
| **Lynx** (github.com/bytedance/lynx) | 直接基于 Wan2.1-14B DiT 的双适配器实现，最接近的参考代码 |
| **IPAdapterWAN** (github.com/kaaskoek232/IPAdapterWAN) | SigLIP2 + Perceiver Resampler 对 Wan2.1 注意力层的动态注入 |
| **InstantX IP-Adapter-SD3** (github.com/instantX-research/IP-Adapter-for-SD3) | DiT 上的 IP-Adapter 基础架构（TimeResampler + adaLN） |
| **HunyuanCustom** (github.com/Tencent-Hunyuan/HunyuanCustom) | 另一个 DiT 视频模型的多模态条件注入 |

---

## 3. 后续工作计划

### 阶段一：IP-Adapter 基础实现与验证

| 步骤 | 任务 | 预期产出 |
|------|------|---------|
| 1 | 分析 Lynx / IPAdapterWAN 源码，理解 DiT 上的图像注入实现细节 | 技术方案文档 |
| 2 | 实现图像编码器模块（CLIP ViT-H/14 + Perceiver Resampler） | image_encoder.py |
| 3 | 改造 DiffSynth-Studio 的 Wan2.1 DiT Cross-Attention 层，新增图像 K/V 投影 | 修改后的 DiT 模块 |
| 4 | 搭建训练管线：冻结 DiT + 编码器，仅训练 Resampler + K/V 投影 | 训练脚本 |
| 5 | 在 MSR-VTT 或小规模数据上验证训练流程跑通 | LoRA + IP-Adapter 权重 |
| 6 | 推理验证：对比 baseline / LoRA / IP-Adapter 的生成效果 | 对比视频 + 评估指标 |

### 阶段二：方案 B 验证（IP-Adapter + VACE/ControlNet）

| 步骤 | 任务 | 预期产出 |
|------|------|---------|
| 7 | 将 IP-Adapter（全局外观引导）与 VACE/ControlNet（空间结构约束）叠加 | 组合条件注入方案 |
| 8 | 在 PVTT 测试集上评估组合方案 vs 单独 VACE 的性能差异 | 对比报告 |
| 9 | 调整 lambda 和 ControlNet 强度，寻找最优平衡点 | 消融实验结果 |

### 阶段三：扩展与优化

| 步骤 | 任务 |
|------|------|
| 10 | 探索 DINOv2 替换 CLIP 作为图像编码器（更强的空间特征） |
| 11 | 从 1.3B 扩展到 14B |
| 12 | 构建产品专用评估指标（产品外观一致性、背景自然度） |

---

## 附录：关键参考文献

| 文献 | 链接 | 关键贡献 |
|------|------|---------|
| IP-Adapter (Ye et al.) | arXiv 2308.06721 | 解耦 Cross-Attention 图像注入的开创性工作 |
| VACE | arXiv 2503.07598 | Wan2.1 官方条件注入框架 |
| Lynx | arXiv 2509.15496 | Wan2.1 上的双适配器身份保持方案 |
| Phantom | arXiv 2502.11079 | 多主体一致性，支持产品类 |
| Saber | arXiv 2512.06905 | Wan2.1 上的零样本 R2V，OpenS2V 最高分 |
| ConsisID | arXiv 2411.17440 | DiT 上的频率分解身份保持 |
| HunyuanCustom | arXiv 2505.04512 | DiT 视频模型多模态条件注入 |
| InstantX IP-Adapter-SD3 | github | DiT 架构上 IP-Adapter 的基础实现 |
