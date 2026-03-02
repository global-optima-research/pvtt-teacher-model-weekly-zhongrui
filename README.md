# week2阶段性工作总结：PVTT 教师模型调研与基线部署

## 1. 学习调研
完成 Wan2.1 架构与 PVTT 技术调研文档（第一、二部分）的学习，明确后续魔改方向：
* **输入层改造**：针对 Source Video + Mask，计划在 VAE 编码后的 Latent 空间（16通道）进行 Channel Concatenation。
* **注意力层改造**：针对 Reference Image，计划参考 IP-Adapter 的解耦机制，在原有 DiT Block 中新增接收图像特征的交叉注意力投影层（Cross-Attention）。

## 2. 实验复现
参考肖同学的《快速走通 Wan 模型 LoRA 微调全流程》，在 RTX 5090 (sm_120 架构) 上完成 Conda + DiffSynth + Wan2.1-1.3B 的环境搭建与微调跑通。

### 2.1 基础环境与框架部署
```bash
# 1. 创建并激活 Conda 环境
conda create -n diffs python=3.10 -y
conda activate diffs

# 2. 安装适配 RTX 5090 (Blackwell 架构) 的稳定版 PyTorch 引擎
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

# 3. 部署 DiffSynth-Studio 框架及依赖
git clone [https://github.com/modelscope/DiffSynth-Studio.git](https://github.com/modelscope/DiffSynth-Studio.git)
cd DiffSynth-Studio
pip install -e .
pip install modelscope accelerate deepspeed peft imageio decord huggingface_hub
```

### 2.2 数据集与模型拉取
```bash
# 回到工作区根目录
cd ~/PVTT_Workspace

# 下载官方样例数据集
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset

# （注：模型权重 Wan-AI/Wan2.1-T2V-1.3B 由训练脚本运行时通过 ModelScope 自动拉取至本地缓存目录 ~/.cache/modelscope/hub/Wan-AI/）
```

### 2.3 启动 LoRA 微调基线
创建并配置 `my_train.sh` 脚本：
```bash
# my_train.sh 文件内容
accelerate launch --mixed_precision="bf16" DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./output" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32
```

分配空闲显卡点火运行：
```bash
chmod +x my_train.sh
CUDA_VISIBLE_DEVICES=0 bash my_train.sh
```

## 3. 辅助工具部署
* **版本控制**：将远程算力服务器的本地工作区接入 GitHub 仓库，保障后续 DiT 架构源码修改的可追溯性与代码安全。
* **AI 编码辅助**：在本地环境部署 ClaudeCode 工具，为下一步编写交叉注意力改造代码与自定义数据加载器（Dataloader）做效能准备。
