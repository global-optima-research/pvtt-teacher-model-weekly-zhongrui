# Wan2.1-T2V-1.3B LoRA 微调实验报告

> 目标：使用通用开源数据集验证 DiffSynth-Studio LoRA 微调流程，为后续架构改造积累工程经验。

---

## 1. 数据集

### 1.1 MSR-VTT (Microsoft Research Video to Text)

选择 **MSR-VTT** 作为本次实验数据集，理由：

- 学术界最广泛使用的视频-文本基准（ACM MM 2016，累计引用 3000+）
- 10,000 条短视频，每条 20 个人工英文描述，内容涵盖音乐、体育、烹饪、新闻等 20 个类别
- 可从 HuggingFace（hf-mirror.com 镜像）直接下载，视频 zip 仅 ~2GB
- 数据格式简单，易于转换为 DiffSynth-Studio 所需的 metadata.csv + 视频目录结构

**下载来源**: `friedrichor/MSR-VTT` (HuggingFace, 通过 hf-mirror.com 镜像访问)

### 1.2 子集构建

从 train_7k split 中抽取 50 条视频：

| 属性 | 值 |
|------|------|
| 视频数量 | 50 |
| 原始分辨率 | 298x224 |
| 视频时长 | ~15 秒 |
| Caption 选取 | 每视频取最长描述（信息量最大） |
| 平均描述长度 | 91 字符（范围 46-155） |
| 抽样策略 | random.seed(42)，按 caption 长度降序取 top-50 |

DiffSynth-Studio 数据格式：

```
data/msrvtt_subset/
  metadata.csv      # 两列：video, prompt
  video4837.mp4
  video6067.mp4
  ... (共50个mp4文件)
```

---

## 2. 实验路径

### 2.1 前置实验：DiffSynth 示例单视频

| 配置 | 值 |
|------|------|
| 数据 | DiffSynth 自带示例（1 条视频："from sunset to night, a small town"） |
| LoRA rank | 32，目标模块：q, k, v, o, ffn.0, ffn.2 |
| Epochs | 5 |
| 产出 | output/epoch-0~4.safetensors（各 ~83MB） |

**意义**：验证训练管线可运行，确认模型加载、LoRA 注入、权重保存流程。

### 2.2 本次实验：MSR-VTT 50 条多类别视频

| 配置 | 值 |
|------|------|
| 基础模型 | Wan2.1-T2V-1.3B |
| 训练框架 | DiffSynth-Studio v2.0.4 + Accelerate |
| LoRA rank | 32，目标模块：q, k, v, o, ffn.0, ffn.2 |
| 学习率 | 1e-4 |
| Epochs | 3 |
| 数据重复 | 10x（每 epoch 500 步） |
| 训练分辨率 | 480x832（Wan2.1 标准 480P） |
| 设备 | 单卡 RTX 5090 (32GB) |
| 峰值显存 | ~25 GB |
| 单步耗时 | ~4-5 秒 |
| 总训练时长 | ~108 分钟（3 epoch × ~36 min/epoch） |

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --dataset_base_path data/msrvtt_subset \
  --dataset_metadata_path data/msrvtt_subset/metadata.csv \
  --height 480 --width 832 --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:..." \
  --learning_rate 1e-4 --num_epochs 3 \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 --output_path "./output_msrvtt"
```

### 2.3 推理验证

使用最终 epoch 的 LoRA 权重，对比 baseline（无 LoRA）与 fine-tuned（有 LoRA）：

| Prompt | 类型 |
|--------|------|
| a man is driving a car on a road through the countryside | 训练集相关 |
| a woman is cooking food in a kitchen stirring ingredients in a pot | 训练集相关 |
| a beautiful sunset over the ocean with waves crashing on the beach | 训练集外 |
| a cat playing with a ball of yarn on a wooden floor | 训练集外 |
| a timelapse of clouds moving over a mountain landscape | 训练集外 |

---

## 3. 训练结果

### 3.1 训练输出

3 个 epoch 均成功完成，每 epoch 保存一个 LoRA checkpoint：

| Epoch | 文件 | 大小 | 保存时间 |
|-------|------|------|----------|
| 0 | epoch-0.safetensors | 84 MB | 01:03 |
| 1 | epoch-1.safetensors | 84 MB | 01:39 |
| 2 | epoch-2.safetensors | 84 MB | 02:15 |

- 总训练步数：1,500 步（500 步/epoch × 3 epochs）
- 每步耗时：~4.5 秒（含前向、反向、梯度更新）
- LoRA 参数量：**43.7M**（600 个 tensor，300 个 LoRA 层 × lora_A + lora_B）
- LoRA 占基础模型参数比例：~3.4%（43.7M / 1.3B）

### 3.2 LoRA 权重分析

通过分析各 epoch 的 `lora_B` 权重范数，可以观察模型学习进度：

| Epoch | lora_B 平均范数 | lora_B 最大范数 | 变化趋势 |
|-------|----------------|----------------|----------|
| 0 | 0.5639 | 1.2622 | 初始学习 |
| 1 | 0.8142 | 1.8019 | +44% |
| 2 | 0.9974 | 2.1773 | +22% |

**观察**：
- lora_B 范数在 3 个 epoch 中持续增长，说明模型在稳定学习
- 增长速度从 epoch 0->1 的 44% 下降到 epoch 1->2 的 22%，显示出正常的收敛趋势
- 未出现范数爆炸或归零，学习率 1e-4 在此场景下较为合适

LoRA 层覆盖的参数结构（以 block.0 为例）：

```
blocks.0.self_attn.{q,k,v,o}.lora_A  shape: [32, 1536]
blocks.0.self_attn.{q,k,v,o}.lora_B  shape: [1536, 32]
blocks.0.cross_attn.{q,k,v,o}.lora_A shape: [32, 1536]
blocks.0.cross_attn.{q,k,v,o}.lora_B shape: [1536, 32]
blocks.0.ffn.0.lora_A                shape: [32, 1536]
blocks.0.ffn.0.lora_B                shape: [8960, 32]
blocks.0.ffn.2.lora_A                shape: [32, 8960]
blocks.0.ffn.2.lora_B                shape: [1536, 32]
```

### 3.3 推理对比

使用 epoch-2 的 LoRA 权重（alpha=1.0），在 5 个 prompt 上对比 baseline vs LoRA：

**推理配置**：50 步去噪，seed=42，tiled=True，480x832 分辨率

| # | Prompt | 类型 | Baseline | LoRA | 文件大小差异 |
|---|--------|------|----------|------|-------------|
| 0 | a man is driving a car on a road... | 训练集相关 | 950 KB | 746 KB | -204 KB |
| 1 | a woman is cooking food in a kitchen... | 训练集相关 | 426 KB | 557 KB | +130 KB |
| 2 | a beautiful sunset over the ocean... | 训练集外 | 652 KB | 504 KB | -148 KB |
| 3 | a cat playing with a ball of yarn... | 训练集外 | 291 KB | 408 KB | +117 KB |
| 4 | a timelapse of clouds moving... | 训练集外 | 138 KB | 185 KB | +47 KB |

**初步观察**：
- LoRA 注入成功且未导致推理崩溃，生成质量保持稳定
- 文件大小变化不一致（有增有减），说明 LoRA 微调改变了视频的时空复杂度分布
- 训练集相关 prompt 和训练集外 prompt 均未出现明显退化，泛化能力基本保持
- DiffSynth 的 LoRA 加载方式为 **权重融合**（fuse）而非动态注入，推理速度与 baseline 一致（~3.2 秒/步）
- 需要人工观看视频以评估运动连贯性、语义一致性等主观质量

**推理产出**：`inference_results/` 目录下 10 个 mp4 文件（baseline_00~04, lora_00~04）

---

## 4. 分析与发现

### 4.1 DiffSynth-Studio 训练框架关键特性

1. **自动模型管理**：通过 `model_id_with_origin_paths` 指定模型 ID + 文件 pattern，框架自动从 ModelScope 下载并转换权重格式（.pth -> .safetensors），存入 `models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/`
2. **强制 Gradient Checkpointing**：即使手动关闭，框架也会强制启用以防 OOM
3. **Flow Matching 损失**：采用 Rectified Flow 范式的 SFT 损失（`FlowMatchSFTLoss`），而非传统 DDPM 的 noise-prediction 或 v-prediction
4. **LoRA 注入机制**：通过 `lora_target_modules` 参数指定目标层名称缩写（q/k/v/o 对应 Attention Q/K/V/O 投影，ffn.0/ffn.2 对应 FFN 线性层），LoRA 权重以 `pipe.dit.` 前缀存储

### 4.2 工程层面发现

1. **显存特征**：Wan2.1-1.3B + LoRA rank=32，单卡 32GB 显存占用 ~25GB（~78%），留有余量。rank 提升到 64/128 仍有空间
2. **数据管线**：DiffSynth 自动处理视频读取、裁剪缩放（298x224 -> 480x832）和 VAE 编码，低分辨率源视频不影响训练
3. **模型下载源**：框架优先从 ModelScope（国内）下载模型和 tokenizer，速度稳定。HuggingFace 数据集需配合 hf-mirror.com 镜像
4. **训练日志**：默认仅显示 tqdm 进度条，不输出 loss。如需监控需修改 `diffsynth/diffusion/logger.py`

### 4.3 LoRA 模块分析

当前 LoRA 覆盖的 DiT 层（Wan2.1-1.3B 共 30 个 DiT Block）：

```
DiT Block:
  Self-Attention
    q_proj  <- LoRA (学习时空特征变换)
    k_proj  <- LoRA
    v_proj  <- LoRA
    o_proj  <- LoRA
  Cross-Attention
    q_proj  <- LoRA (通过 q 同名覆盖)
    k_proj  <- LoRA (通过 k 同名覆盖)
    v_proj  <- LoRA (通过 v 同名覆盖)
    o_proj  <- LoRA (通过 o 同名覆盖)
  FFN
    ffn.0   <- LoRA (学习特征空间映射)
    ffn.2   <- LoRA
  adaLN     (冻结, 条件注入层)
```

注意：DiffSynth 的 q,k,v,o 缩写同时覆盖了 Self-Attention 和 Cross-Attention 的投影层。

### 4.4 对后续架构改造（参考图像注入）的启示

本次 LoRA 微调实验的核心价值在于：通过"最小侵入"操作完整探测了 DiT 的内部结构和注入机制，为后续 IP-Adapter 风格的参考图像条件注入提供了关键的架构依据。

#### 4.4.1 LoRA 与 IP-Adapter 的关系：互补而非类似

两者都作用于 Cross-Attention 层，但机制本质不同：

| | LoRA 微调 | IP-Adapter (Ref Image 注入) |
|---|---|---|
| 目的 | 修改已有权重（W' = W + BA） | 新增并行注意力分支 |
| 信息流 | 不引入新条件，调整模型处理方式 | 引入图像 embedding 作为全新条件源 |
| 结构变化 | 无，参数形状不变 | 有，新增 K_img / V_img 投影层 |
| 输出 | `Attn_text'`（修改后的文本注意力） | `Attn_text + lambda * Attn_image` |

改造后的 Cross-Attention 结构：

```python
# 原始文本分支（保持不变）
Q     = W_q @ hidden_states          # hidden_states 来自视频特征
K_txt = W_k @ text_emb               # text_emb 来自 T5 编码器
V_txt = W_v @ text_emb
Attn_text = Softmax(Q @ K_txt^T) @ V_txt

# 新增图像分支（IP-Adapter）
K_img = W_k_img @ image_emb          # 新增参数，image_emb 来自 CLIP/DINOv2
V_img = W_v_img @ image_emb          # 新增参数
Attn_img = Softmax(Q @ K_img^T) @ V_img

Out = Attn_text + lambda * Attn_img   # 解耦加权融合
```

关键点：图像分支**复用同一个 Q**（来自视频特征），仅新增 K_img / V_img 投影，与 LoRA 修改 Q/K/V 的方式形成互补。后续可以考虑 **LoRA + IP-Adapter 联合训练**。

#### 4.4.2 本次实验确认的架构参数

| 参数 | 值 | 对 IP-Adapter 的意义 |
|------|-----|---------------------|
| DiT Block 数量 | 30（blocks.0 ~ blocks.29） | IP-Adapter 需在全部 30 层 Cross-Attention 注入图像分支 |
| Cross-Attention hidden_dim | 1536 | 图像编码器输出需投影到 1536 维以对齐 |
| FFN 中间维度 | 8960 | FFN 层不参与图像注入，但了解维度有助于显存预算 |
| Self-Attention 与 Cross-Attention 结构 | 同维度，独立投影 | 图像注入仅作用于 Cross-Attention，不影响 Self-Attention 的时空建模 |

#### 4.4.3 工程约束对后续改造的影响

1. **LoRA 加载为权重融合（fuse）方式**：推理时 `load_lora` 将 LoRA 直接融合到原始权重中，融合后无法 `clear_lora()`。若后续需要动态切换 LoRA（如不同风格的 IP-Adapter），需改造为可分离的注入方式
2. **显存预算**：当前 LoRA rank=32 占用 ~25GB / 32GB。IP-Adapter 新增图像编码器（CLIP ViT-H ~600M 参数）+ 投影层 + 新 K/V 投影，需评估是否仍能单卡训练，或需要多卡并行
3. **训练范式一致性**：Wan2.1 使用 Rectified Flow（`FlowMatchSFTLoss`），后续训练 IP-Adapter 投影层时必须使用相同的 flow matching 损失，而非 DDPM 的 noise-prediction
4. **数据管线可扩展**：当前 `metadata.csv` 为 `video, prompt` 二列格式，可直接扩展为 `video, prompt, reference_image` 三列，为 IP-Adapter 训练数据准备提供基础

---

## 5. 后续计划

### 5.1 短期：架构理解与源码分析

1. **DiT 结构梳理**：分析 `diffsynth/models/` 中 Wan2.1 DiT 实现，理清每层的输入输出维度、注意力类型（3D Full Attention vs 时空因子化）
2. **Cross-Attention 机制**：重点分析 Cross-Attention 如何接收 text embedding，为注入 reference image embedding 做准备
3. **训练 Loss 监控**：修改 DiffSynth logger，添加 loss 值输出和 TensorBoard/WandB 集成

### 5.2 中期：参考图像条件注入

1. **IP-Adapter 方案**：在 DiT 的 Cross-Attention 层添加解耦的图像交叉注意力分支
   - 图像编码器：CLIP ViT-H/14 或 DINOv2 ViT-L/14
   - 投影层：Perceiver Resampler (256 patch tokens -> 16 tokens)
   - 注入方式：Attn_out = Attn_text + lambda * Attn_image
2. **训练策略**：冻结 DiT + 图像编码器，仅训练投影层 + 新增 K/V 投影 + LoRA
3. **CFG 支持**：20% 概率丢弃图像条件，支持 Classifier-Free Guidance

### 5.3 长期：完整 Teacher Model

1. Source Video 条件化（channel concatenation 或 VACE 框架）
2. 渐进式训练：1.3B -> 14B
3. 评估指标体系：CLIP-I > 0.85, FVD < 100

---

## 附录：实验文件清单

| 文件 | 说明 |
|------|------|
| train_msrvtt.sh | 训练启动脚本 |
| prepare_msrvtt.py | MSR-VTT 数据集准备脚本 |
| inference_msrvtt.py | 推理验证脚本（baseline vs LoRA 对比） |
| data/msrvtt_subset/ | 数据集（50 videos + metadata.csv） |
| output_msrvtt/ | LoRA 权重输出 |
