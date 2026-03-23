# PVTT 商品视频迁移实验报告 v2

**项目**：Product Video Transfer Task (PVTT)
**实验周期**：2026-03
**负责人**：zhongrui

---

## 一、任务目标

给定三个输入：
- **Source video A**：包含商品 A 的视频（提供运动轨迹、场景背景）
- **Reference image B**：目标商品 B 的静态图（提供外观、纹理、颜色）
- **Target prompt**：描述商品 B 的文本

目标输出：保留 A 的运动与背景结构、呈现 B 的外观的新视频。

---

## 二、已完成的架构改造

### 2.1 整体方案

在 Wan2.1-T2V-1.3B 的基础上，引入两路额外条件：

```
source video A ──VAE──► VaceWanModel ──► 结构 hints ──► DiT 每个 block（加法注入）
ref image B ──CLIP──► PerceiverResampler ──► ip_tokens ──► 并联 Cross-Attention
text prompt ──T5──────────────────────────────────────► 原始 Cross-Attention
                                                                    ↓
                                                            生成视频
```

### 2.2 IP-Adapter（Reference Image 条件注入）

**改造位置**：`wan_video_dit.py` 的 CrossAttention 层 + `wan_video.py` 的推理函数。

每个 DiT block 的 Cross-Attention 新增并联的图像注意力分支：

```
Out = Attn_text(Q, K_text, V_text) + λ × Attn_ip(Q, K_ip, V_ip)
```

- 新增参数：k_ip、v_ip（线性投影）、norm_k_ip（RMSNorm），共 30 个 block 均注入
- λ = ip_scale，推理时可调节图像条件强度

**Reference Image 编码器（PerceiverResampler）**：

```
CLIP ViT-H/14 → 257 tokens (1280-dim) → Perceiver (4层, 16 queries) → 16 tokens (1536-dim)
```

CLIP 权重冻结，Resampler 可训练。

**可训练参数总量**：约 231M（Resampler 90M + 30 × cross-attn 141M），其余（DiT 1.3B、CLIP、T5）全部冻结。

### 2.3 VACE（Source Video 结构条件）

使用预训练的 Wan2.1-VACE-1.3B 权重（735M 参数，无需额外训练）：

- 将源视频帧经 VAE 编码后送入 VaceWanModel，输出每个 DiT block 对应的结构 hint
- 注入方式：`x = x + hint × vace_scale`（各 block 加法，不改变 DiT 结构）
- VACE 权重从 6.7GB safetensors 中提取 vace_blocks.* 和 vace_patch_embedding.* 共 439 个 key，直接加载，无权重缺失

---

## 三、训练过程

### 3.1 训练数据构建

**自监督配对策略**：将每条视频与该视频所展示的商品的参考图配对。

```
训练样本 = (商品 A 的视频, 商品 A 的参考图, 描述商品 A 的 prompt)
```

- 数据来源：PVTT 评估集，53 个视频，8 个商品品类
- 自监督逻辑：模型学习"看到商品 A 的参考图 → 重建商品 A 的视频"，
  推理时将参考图换成商品 B，期望迁移 B 的外观

### 3.2 训练配置

| 参数 | 值 |
|------|----|
| Epochs | 10 |
| Batch size | 1 |
| Learning rate | 1e-4（AdamW） |
| 训练分辨率 | 240×416，9帧 |
| GPU | RTX 5090 × 1（32GB） |
| CFG image dropout | 10%（ip_tokens 置零，保留梯度路径） |

### 3.3 训练结果

| Epoch | avg_loss |
|-------|----------|
| 1 | 0.9205 |
| 2 | 0.6003 |
| 5 | 0.2005 |
| 7 | 0.1746（最低） |
| 10 | 0.1810 |

Loss 稳定收敛，Checkpoint 保存在 `checkpoints/ip_adapter_epoch10.pth`（443MB）。

---

## 四、推理验证与结果分析

### 4.1 实验一：T2V + IP-Adapter（无源视频条件）

对 8 个品类各做一次推理，与纯文本基线对比，输出 HTML 可视化页面。

**发现**：生成视频完全没有 source video A 的结构。原因是 T2V + IP-Adapter 的信号流中，source video A 根本没有输入路径，模型是从随机噪声出发，仅凭 text + ref image B 生成新视频，不是商品替换，是从零生成。

### 4.2 实验二：VACE + IP-Adapter 零样本组合

为解决 source video 无输入路径的问题，引入预训练 VACE 模块做零样本组合。

**测试用例**：
- Source：handfan1.mp4（木柄手扇，有人手持，户外场景）
- Ref：handfan_2.jpg（紫色樱花图案手扇）
- Mask：全 1（全帧可编辑）

**结果**（`vace_ip_test_v3.mp4`）：

| 维度 | 现象 | 原因 |
|------|------|------|
| 扇子形状/角度 | 与源视频大体一致 | VACE 结构 hint 有效 |
| 颜色 | 整体偏向紫色调 | IP-Adapter 外观条件有效 |
| 背景 | 变为粉紫梦幻风格，失真严重 | 全 1 mask 导致背景也被改写 |
| 整体清晰度 | 模糊，细节丢失 | 见下方"当前局限"分析 |

---

## 五、当前局限与根本原因

### 局限 1：训练与推理存在分布不匹配（导致图像模糊的直接原因）

IP-Adapter 训练时，DiT 的输入是纯 T2V 的 hidden states。但 VACE+IP 零样本推理时，VACE 对 DiT 每个 block 的 hidden states 做了强力加法修改：

```
训练时：DiT block 输入 x（正常分布）→ IP cross-attn 投影 → 输出
推理时：DiT block 输入 x + VACE_hint（从未见过的分布）→ IP cross-attn 投影 → 输出崩溃
```

IP-Adapter 的 k_ip / v_ip 是在无 VACE 的 hidden states 上学到的，当输入分布改变后，attention 计算结果失效，导致生成物模糊。**VACE 和 IP-Adapter 从未联合训练，零样本组合的上限就在这里。**

### 局限 2：训练数据量严重不足

| 方案 | 训练样本量 |
|------|-----------|
| 原版 IP-Adapter（图像） | ~100 万对 |
| Video IP-Adapter（通常） | 数十万对 |
| 本次训练 | **53 对** |

53 个样本无法让 231M 参数的模型学会跨商品泛化的外观迁移能力。

### 局限 3：全 1 Mask 无背景保留

VACE 的 mask=0 区域对应"保留源视频像素"，mask=1 对应"允许模型改写"。当前使用全 1 mask，背景和商品区域都被改写，导致背景失真。

---

## 六、后续工作计划

### Phase 1：接入 SAM 2 自动生成商品 Mask（工程增强，无需训练）

**目标**：让背景完全保留，仅商品区域被 IP-Adapter 引导改写。

| 步骤 | 工具 |
|------|------|
| 文本驱动首帧商品检测 | Grounding DINO |
| 首帧精细分割 | SAM 2 |
| 全视频 mask 传播 | SAM 2 Video Predictor |
| 接入 VACE：inactive=背景帧, reactive=商品帧 | 现有推理脚本改造 |

这一步直接将当前"全帧风格迁移"升级为"精确商品区域替换"，与后续数据集格式（source video + mask + ref image）完全对齐。

### Phase 2：基于标注数据集的联合训练（解决根本问题）

后续数据集格式：`source video A + product mask + reference image B + GT video`

有了 GT video 后，训练范式从自监督升级为有监督：

```
训练目标（当前自监督）：           训练目标（有监督，正确）：
video A → 噪声目标                GT video（商品B外观）→ 噪声目标
ref image A → IP 条件             ref image B → IP 条件
                                  source video A + mask → VACE 条件（联合训练）
```

训练策略：
- IP-Adapter（k_ip/v_ip/Resampler）继续训练，从当前 checkpoint 热启动
- VACE 模块（vace_patch_embedding）解冻，参与梯度更新
- Loss 在商品 mask 区域加权，聚焦外观迁移质量

### Phase 3：系统评估

- 在完整评估集（199 任务）运行 ip_scale 消融（0.5 / 0.75 / 1.0 / 1.25 / 1.5）
- 评估指标：CLIP-I2V（ref image 与生成视频外观相似度）、Temporal Consistency、LPIPS（与 GT 对比）

---

## 七、关键文件

```
ip_adapter/
  scripts/
    train_ip_adapter.py    # 训练主程序
    infer_vace_ip.py       # VACE + IP-Adapter 零样本推理（当前最新）
    run_demo_html.py       # 8品类对比 HTML 可视化
  checkpoints/
    ip_adapter_epoch10.pth # 最终 checkpoint（443MB）
  output/
    demo_result.html       # T2V+IP 8品类对比可视化
    vace_ip_test_v3.mp4    # VACE+IP 零样本测试结果

DiffSynth-Studio/diffsynth/models/
  wan_video_dit.py         # CrossAttention + DiTBlock（已改造）
  wan_video.py             # model_fn ip_tokens 透传（已改造）

models/Wan-AI/
  Wan2.1-T2V-1.3B/         # 基础 DiT
  Wan2.1-VACE-1.3B/        # VACE 权重（6.7GB）
  clip/                    # CLIP ViT-H/14
```
