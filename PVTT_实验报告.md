# PVTT 商品视频迁移实验报告

**项目**：Product Video Transfer Task (PVTT)
**实验周期**：2026-03
**负责人**：zhongrui
**服务器**：111.17.197.107 · `/data/zhongrui/PVTT_Workspace`

---

## 一、实验目标

给定：
- **Source video A**：包含商品 A 的原始视频（运动、背景、光照）
- **Reference image B**：目标商品 B 的静态图像（外观、纹理、颜色）
- **Target prompt**：描述商品 B 的文本

要求生成：保留 A 的运动轨迹与背景结构、呈现 B 的外观的新视频。

原始任务拆解对应完成情况：

| 子任务 | 状态 |
|--------|------|
| Wan2.1 DiT 架构分析 | ✅ 完成 |
| Reference Image 编码器 | ✅ 完成 |
| Cross-Attention 条件注入 | ✅ 完成并训练 |
| Source Video 条件化 | ⚠️ 零样本方案替代（未训练） |
| 架构验证 | ✅ 完成 |

---

## 二、架构改造

### 2.1 基础模型

- **Backbone**：Wan2.1-T2V-1.3B（DiT，30个 WanAttentionBlock，dim=1536）
- **框架**：DiffSynth-Studio
- **推理分辨率**：480×832，训练/demo 帧数 9–49 帧

### 2.2 IP-Adapter 注入（Cross-Attention 改造）

在 DiffSynth-Studio 的 `wan_video_dit.py` 和 `wan_video.py` 中进行了以下改造：

**CrossAttention 层**（每个 DiT block 的 cross-attn）新增三组可训练投影：
```python
self.k_ip     = nn.Linear(dim, dim, bias=True)   # 图像 key
self.v_ip     = nn.Linear(dim, dim, bias=True)   # 图像 value
self.norm_k_ip = RMSNorm(head_dim, eps=1e-6)      # key 归一化
```

注入方式为**解耦并联 Cross-Attention**（Decoupled Parallel Cross-Attention）：
```
Out = Attn_text(Q, K_text, V_text) + λ × Attn_ip(Q, K_ip, V_ip)
```
其中 λ = `ip_scale`（推理时可调，训练时固定为 1.0）。所有 30 个 block 均注入，与文本 cross-attn 并联，互不干扰。

**DiTBlock** 新增 `ip_tokens` 参数透传，`model_fn_wan_video` 新增 `ip_tokens` kwarg 转发至每个 block 的 cross-attn。

### 2.3 PerceiverResampler（Reference Image 编码器）

```
CLIP ViT-H/14 (1280-dim, 257 tokens) → PerceiverResampler → (B, 16, 1536)
```

- 4 层 Perceiver，16 个可学习 query，dim=1536（与 DiT 一致）
- CLIP 模型：`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`（多语言 ViT-H/14）
- CLIP 权重**冻结**，仅 Resampler 可训练

### 2.4 可训练参数统计

| 模块 | 参数量 |
|------|--------|
| PerceiverResampler | ~90M |
| k_ip / v_ip / norm_k_ip × 30 blocks | ~141M |
| **合计可训练** | **~231M** |
| DiT 1.3B（冻结） | 1300M |
| CLIP ViT-H（冻结） | ~1800M |
| T5 文本编码器（冻结） | ~4500M |

---

## 三、训练

### 3.1 训练数据

- 数据集：PVTT 评估集（53 个视频，35 张商品图，199 个任务）
- 训练样本：53 条（video + ref_image + target_prompt）
- 品类：手扇、手袋、项链、钱包、太阳镜、手表等 8 个品类

数据处理流程：
```
source video → imageio 抽帧 → VAE 编码 → noise latent（训练目标）
ref image → CLIP visual (use_31_block=True) → PerceiverResampler → ip_tokens
target_prompt → T5 → text_tokens
```

### 3.2 训练配置

| 参数 | 值 |
|------|----|
| Epochs | 10 |
| Batch size | 1 |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Grad clip | max_norm=1.0 |
| CFG dropout | 10%（随机将 ip_tokens 置零，保留计算图） |
| 训练帧数 | 9 帧 |
| 分辨率 | 240×416（显存限制） |
| GPU | RTX 5090 × 1（32GB） |
| 框架 | 纯 PyTorch（无 accelerate） |

**注意**：未使用 accelerate mixed_precision，直接以 `bfloat16` 运行。CFG dropout 使用 `torch.zeros_like(ip_tokens)` 而非 `ip_tokens=None`，以确保反向传播梯度路径完整。

### 3.3 训练曲线

| Epoch | avg_loss |
|-------|----------|
| 1 | 0.9205 |
| 2 | 0.6003 |
| 4 | 0.2647 |
| 5 | 0.2005 |
| 6 | 0.2105 |
| 7 | 0.1746 |
| 8 | 0.1895 |
| 10 | 0.1810 |

Loss 从 0.92 下降至 0.18，收敛稳定。Epoch 7 达到最低点 0.1746。

**Checkpoint**：`ip_adapter/checkpoints/ip_adapter_epoch10.pth`（443MB，含 resampler + 30×cross_attn）

---

## 四、推理验证

### 4.1 T2V + IP-Adapter 推理（Demo HTML）

在 8 个品类各选 1 个任务，对比 IP-Adapter 生成 vs 纯 T2V 基线：

**发现**：生成结果忽略了 source video A。架构缺陷在于 T2V + IP-Adapter 的输入路径只有：
```
text + ref image B → 从噪声生成新视频
source video A：❌ 无任何输入路径
```
生成物是"基于 B 的外观从零生成的视频"，不是商品替换。

### 4.2 VACE + IP-Adapter 零样本推理

引入 Wan2.1-VACE-1.3B 的预训练 VaceWanModel 作为 source video 条件化模块：

**信号流**：
```
source video A ──VAE──► VaceWanModel ──► hints[i]
                                              ↓ x = x + hints[i] × vace_scale
ref image B ──CLIP──Resampler──► ip_tokens ──► parallel cross-attn
text prompt ──T5──────────────────────────► text cross-attn
                                              ↓
                                        生成视频
```

**VACE 权重加载**：从 VACE-1.3B safetensors（6.7GB）中提取 `vace_blocks.*` 和 `vace_patch_embedding.*` 共 439 个 key，直接加载到 VaceWanModel（735M params），无 key 不匹配。

**实验参数**：
- Source: `0001-handfan1.mp4`（原始手扇视频）
- Ref: `handfan_2.jpg`（紫色樱花图案手扇）
- Prompt: "A purple fabric hand fan with cherry blossom pattern, waving gently in the breeze"
- Mask：全 1（fully reactive，无背景保留）
- Steps: 20，Frames: 9

**结果**（见 `output/vace_ip_test_v3.mp4`）：

| 维度 | 结果 |
|------|------|
| 扇子形状/角度 | ✅ 与源视频结构一致（VACE hint 有效） |
| 颜色/外观 | ✅ 向紫色/薰衣草偏移（IP-Adapter 有效） |
| 背景 | ❌ 变为粉紫梦幻风格（全 1 mask 无背景保留） |
| 整体质量 | 偏艺术感，非干净商品替换 |

---

## 五、当前方案的局限

### 5.1 全 1 Mask 无法保留背景

VACE 的 mask 语义：
- `mask=0`（inactive）：该像素从源视频直接保留，不做生成
- `mask=1`（reactive）：该像素由模型生成，受 IP-Adapter 引导改写

当前使用全 1 mask，背景也被 IP-Adapter 外观条件影响，导致背景颜色被染色。

### 5.2 IP-Adapter 训练数据缺少 Source Video 信号

现有训练只优化了"ref image B → 视频外观"的映射。训练中 source video 经过 VAE 编码但仅作为扩散噪声目标，VACE hint 并未参与训练过程。即：**VACE 和 IP-Adapter 从未联合训练，零样本组合效果有上限**。

### 5.3 训练数据规模

53 条样本极少，IP-Adapter 的泛化能力有限。

---

## 六、后续工作计划

### Phase 1：商品 Mask 自动生成（解决背景保留问题）

**目标**：为每帧生成商品区域 mask，实现真正意义上的商品替换。

| 步骤 | 工具 | 产出 |
|------|------|------|
| 首帧商品检测 | Grounding DINO（文本驱动检测） | 商品 BBox |
| 首帧精细分割 | SAM 2（Segment Anything） | 首帧 mask |
| 全帧 mask 传播 | SAM 2 video predictor | 全视频 mask 序列 |
| VACE 双通道编码 | `inactive=背景帧, reactive=商品帧` | 结构+保留信号 |

这一步无需额外训练，是对现有推理流程的增强。

**预期效果**：背景完全保留，仅商品区域被 IP-Adapter 引导改写外观。

### Phase 2：VACE + IP-Adapter 联合训练

**目标**：让模型学会在 VACE 结构约束下，正确将 ref image 外观迁移到商品区域。

需要真实标注数据（source video A + product mask + ref image B + GT video），即后续数据集的格式。

训练策略调整：
- 将 VACE module 的 `vace_patch_embedding` 解冻，允许微调
- 使用真实 mask 替代全 1 mask
- loss 仅计算商品区域（mask 加权 loss）

### Phase 3：扩展评估

- 在完整 PVTT 评估集（199 任务）上运行 ip_scale 消融实验（0.5 / 0.75 / 1.0 / 1.25 / 1.5）
- 评估指标：CLIP-I2V（外观相似度）、Temporal Consistency、LPIPS（与 GT 对比）
- 与纯 T2V 基线对比，验证 IP-Adapter 的增益

---

## 七、关键文件索引

```
ip_adapter/
  scripts/
    train_ip_adapter.py     # 训练主程序（纯 PyTorch，含 inject_ip_adapter）
    infer_ip_adapter.py     # T2V + IP-Adapter 推理
    infer_vace_ip.py        # VACE + IP-Adapter 零样本推理 ← 当前最新
    run_demo_html.py        # 8品类对比 HTML 可视化
    batch_infer.py          # 批量推理脚本
    eval_metrics.py         # CLIP-I2V / 时序一致性评估
  src/
    resampler.py            # PerceiverResampler
  checkpoints/
    ip_adapter_epoch10.pth  # 最终 checkpoint（443MB）
  output/
    demo_result.html        # 8品类推理对比（T2V+IP vs 基线）
    vace_ip_test_v3.mp4     # VACE+IP 零样本测试结果

DiffSynth-Studio/diffsynth/models/
  wan_video_dit.py          # CrossAttention + DiTBlock 改造（已 patch）
  wan_video.py              # model_fn_wan_video ip_tokens 透传（已 patch）

models/Wan-AI/
  Wan2.1-T2V-1.3B/          # 基础 DiT 权重
  Wan2.1-VACE-1.3B/         # VACE 权重（6.7GB，含 vace_blocks.* 439 keys）
  clip/                     # CLIP ViT-H/14 多语言版
```
