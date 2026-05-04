# PVTT FFGo 5B 一周实验计划

---

## 资源盘点（当前状态）

| 资源 | 路径 | 状态 |
|------|------|------|
| PVTT 训练数据集（主） | `/data/datasets/ffgo_datasets/ffgo_training_data_480x832/` | ✅ 330 样本，832×480，Grounding-DINO+SAM2+ProPainter |
| PVTT 训练数据集（新） | `/data/datasets/ffgo_datasets/ffgo_dataset_20260409/` | ✅ 283 样本，2026-04-09 版本 |
| Wan2.2 TI2V 5B 底模 | 服务器上未找到（hf /ms下载） | ❌ 待补充 |
| xiaohongrui week07 LoRA checkpoint | `/data/xiaohongrui/` 目录为空（无访问权限） | ❌ 待补充 |
| 评估数据集 | 待提供 （our benchmark） | ❌ 待补充 |

---

## 两个方向的论点与实验设计

### 方向一：5B vs 14B baseline——为什么 5B 对 PVTT 任务更适配

#### 核心论点

| 维度 | FFGo 原版（14B MoE，I2V-A14B） | 我们的方案（5B Dense，TI2V-5B） |
|------|------|------|
| 架构 | Mixture-of-Experts，专家路由稀疏激活 | Dense，全参数参与，LoRA 梯度流更稳定 |
| 训练框架 | VideoX-Fun（为 MoE 定制，适配成本高） | DiffSynth-Studio（原生支持 TI2V-5B） |
| 显存（LoRA rank=128） | >32GB，需多卡并行 | ~24-28GB，单张 RTX 5090 可完成训练 |
| 输入形式 | I2V（参考图+文本） | TI2V（文本+首帧拼贴图），与 FFGo 范式天然契合 |
| 领域适配 | 通用电商视频，无 PVTT 专项数据 | 在 330 条 PVTT 数据上微调，domain-specific |
| 训练效率 | ~50 epochs 需多卡，周期长 | ~50 epochs 单卡，约 2-3 天 |

#### 需要的实验

- 用 hongrui week07 LoRA checkpoint 在 PVTT 评估集上批量推理
- 与已有参照结果对比（三档）：
  - 下界：零样本 VACE 1.3B（上周实验已有）
  - 对比：FFGo 14B 论文公开指标（MFS=0.992, ProdCLIP=0.730）
  - 目标：FFGo 5B LoRA（本周测量）
- 指标：MFS（帧间一致性）、ProdCLIP（产品匹配度） -> **我们的benchmark**

---

### 方向二：FFGo 5B vs 其他视频编辑方法的优势

#### 方法对比矩阵

| 方法 | 核心范式 | 产品完整替换 | 产品身份保持 | PVTT 适配性 | 备注 |
|------|------|:---:|:---:|:---:|------|
| **TokenFlow** | Token 传播，保结构 | ✗ | 无显式机制 | 差 | 只能纹理级编辑，无法跨类别换产品 |
| **AnyV2V** | 编辑首帧后传播全帧 | △ | 间接，随帧衰减 | 中 | 依赖图像编辑质量，难以完全替换产品 |
| **WanEdit** | 指令驱动视频编辑 | △ | 无视觉参考 | 中 | 仅靠 prompt，无法精确控制产品外观 |
| **FFGo 5B（ours）** | 首帧拼贴 + TI2V LoRA | ✅ | 显式首帧约束 | 强 | 完整视觉参考 + 转场生成，领域专项微调 |

#### FFGo 5B 的核心优势

1. **完整产品替换而非编辑**：TokenFlow/AnyV2V 基于源视频结构做局部修改，本质是"编辑"；FFGo 通过转场生成全新视频，实现跨类别产品"替换"
2. **显式视觉参考**：首帧左半直接放置目标产品图，身份约束在像素级而非语义级，优于仅靠 prompt 的 WanEdit
3. **领域专项 LoRA**：在 PVTT 数据集（电商产品视频）上微调，模型学习了产品展示的运动模式，通用方法不具备

#### 需要的实验

- **在相同 3-5 条 PVTT 评估任务上**，与各方法对比：
  - 定量：MFS、ProdCLIP、SSIM
  - 定性：4 种方法结果并排可视化
- **任务选择策略**：覆盖不同难度
  - 形状相似（同类产品替换，如手扇→手扇）
  - 形状差异较大（跨类别，如手扇→手表）

---

## 一周时间轴

```
Day 1（Mon）: 准备阶段
  - 确认 Wan2.2 TI2V 5B 底模和 week07 LoRA checkpoint 服务器路径
  - cp -r /data/datasets/ffgo_datasets/ffgo_training_data_480x832 /data/zhongrui/（只读副本）
  - 参照 training_guide 搭建推理脚本，跑通单条推理

Day 2（Tue）: FFGo 5B 批量推理
  - 在评估集上批量推理（nohup，GPU 5）
  - 输出：每条任务的生成视频 + 转场裁剪结果

Day 3（Wed）: 定量评估
  - 计算评估集上 MFS / ProdCLIP 指标
  - 汇总三档对比表格（VACE 1.3B 零样本 / FFGo 14B 论文 / FFGo 5B 实测）

Day 4（Thu）: 竞品对比实验
  - 选 3-5 条任务跑 AnyV2V / WanEdit（或调用服务器已有推理脚本）
  - 无法运行时用公开 benchmark 数字补充
  - 制作定性对比可视化（4 方法并排）

Day 5（Fri）: 整理 + 周报
  - 汇总定量表格
  - 输出可视化图
  - 填写周报
```

---

## 关键参数备忘（来自 training_guide）

```
底模：    Wan2.2 TI2V 5B (Dense)
框架：    DiffSynth-Studio
LoRA：    rank=128, lr=1e-4, weight_decay=3e-2
目标层：  q,k,v,o,ffn.0,ffn.2
分辨率：  832×480，81 帧，16fps
触发词：  "ad23r2 the camera view suddenly changes"
显存：    ~24-28GB（RTX 5090）
推理：    inference_ffgo_lora.py --product_image [RGBA] --background_image [bg] --lora_path [ckpt]
          自动丢弃前 4 帧转场
```

## 数据集备忘

```
主数据集：/data/datasets/ffgo_datasets/ffgo_training_data_480x832/
  - 330 个样本（sample_000 ~ sample_329）
  - 每个样本：first_frame.png + video.mp4 + caption.txt + metadata.json
  - 产品提取：Grounding-DINO + SAM2（精确，优于 rembg）
  - 背景修复：ProPainter（视频级 inpainting，优于 LaMa）
  - caption：Qwen2-VL-7B-Instruct

新数据集：/data/datasets/ffgo_datasets/ffgo_dataset_20260409/（283 样本）
  - 注意：不要直接修改，使用前 cp 到 /data/zhongrui/
```
