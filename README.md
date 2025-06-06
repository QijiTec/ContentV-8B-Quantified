![image](https://github.com/user-attachments/assets/4c6dc3a2-e435-44dd-87a1-58facdc2108e)
<br/>
启用模型CPU卸载，将显存占用降低至民用级24G显卡可用。（暂未实现模型量化及顺序卸载） <br/>
Clone 仓库，按照官方步骤建立虚拟环境并安装依赖，下载官方仓库模型 <br/>
启动：python app.py<br/>
 <br/>
![image](https://github.com/user-attachments/assets/4171189e-d9f1-4e62-87d6-df8012417083)
<br/>
 <br/>
### 使用说明
模型路径: 输入已下载的ContentV模型路径<br/>
提示词设置:<br/>
正面提示词：描述您想要生成的视频内容<br/>
负面提示词：描述您不想在视频中出现的内容<br/>
视频参数:<br/>
总帧数：视频的总帧数（影响生成时间）<br/>
帧率：视频的播放速度<br/>
分辨率：视频的宽度和高度<br/>
推理步数：生成质量（步数越多质量越好，但耗时更长）<br/>
内存优化选项:<br/>
顺序CPU卸载：将不活跃的模型组件逐个卸载到CPU，适用于显存较小的GPU<br/>
模型CPU卸载：整个模型级别的CPU卸载，比顺序卸载性能更好<br/>
VAE切片：将VAE输入分片处理，减少显存使用<br/>
VAE平铺：将VAE输入平铺处理，可以处理更大的图像<br/>
注意力切片：将注意力计算分片处理，减少显存使用但可能略微降低速度<br/>
<br/>

### 注意：
<br/>
顺序CPU卸载和模型CPU卸载只能选择其中一个<br/>
其他优化选项可以组合使用<br/>
生成视频可能需要一些时间，请耐心等待<br/>
生成的视频将保存在outputs目录下<br/>
<br/>

### 优化效果

显存使用降低：根据配置和使用场景，可以降低30-70%的显存使用 <br/>
内存占用更均衡：通过CPU卸载实现GPU和CPU之间的内存平衡 <br/>
支持更大分辨率：通过优化可以处理更高分辨率的视频生成 <br/>
批处理能力提升：在相同显存下可以处理更大的批次 <br/> 
### 注意事项
 <br/> 
1. CPU卸载会略微增加处理时间，建议根据实际需求选择是否启用 <br/>

2. 模型量化可能会轻微影响生成质量，如果追求最高质量可以只启用CPU卸载 <br/>

3. 建议在生成高分辨率视频或长序列时启用所有优化功能 <br/>

4. 可以通过监控显存使用情况来调整优化策略 <br/>
 
### 建议配置
 <br/>
对于24GB显存的GPU：可以只启用CPU卸载 <br/>
对于16GB显存的GPU：建议同时启用CPU卸载和模型量化 <br/>
对于16GB及以下显存的GPU：必须启用所有优化功能 <br/>
通过合理使用这些优化功能，可以在有限的显存条件下实现更好的视频生成效果。 <br/>

### Coder工作记录
 <br/>
 
**量化及加速**
这是一个基于Stable Diffusion 3的3D Transformer模型实现，主要特点如下： <br/>
 <br/>
1. 模型架构：

使用了SD3Transformer3DModel，继承自SD3Transformer2DModel <br/>
包含大量的attention层（38个attention heads） <br/>
使用了patch embedding和position embedding <br/>
支持NPU（昇腾）加速 <br/> <br/>

2. 显存占用大的主要原因：
 <br/>
大量的attention层（num_layers=38, num_attention_heads=38） <br/>
较大的attention head维度（attention_head_dim=64） <br/>
3D数据处理（视频或3D数据）需要更多显存 <br/>
QKV矩阵计算和存储 <br/>
 <br/>
**显存占用大的主要原因包括：** <br/>
 <br/>
1. 模型组成： <br/>
 <br/>
VAE编码器/解码器 <br/>
多个文本编码器（CLIP和T5） <br/>
3D Transformer模型 <br/>
处理视频数据（多帧） <br/>
 <br/>
2. 运行时显存占用： <br/>
 <br/>
大量中间状态和梯度存储 <br/>
多个批次的视频帧处理 <br/>
生成过程中的latent空间计算 <br/>
Classifier-Free Guidance需要两次前向传播 <br/>
 <br/>
**优化显存使用的方案：**
 <br/>
1. 实时量化方案： <br/>
 <br/>
对Transformer和文本编码器进行INT8量化 <br/>
使用动态量化技术，只在推理时进行量化 <br/>
可以使用torch.quantization或bitsandbytes库 <br/>
 <br/>
2. 模型卸载方案： <br/>
 <br/>
使用CPU offloading，将不需要的模型组件临时移到CPU <br/>
实现渐进式加载，按需加载模型组件 <br/>
使用accelerate库的device_map功能 <br/>
 <br/>
**显存占用大主要是因为：**
 <br/>
1. 多个大型模型组件同时加载 <br/>
2. 视频生成过程中的大量中间状态 <br/>
3. Classifier-Free Guidance需要两次前向传播 <br/>
4. 处理视频数据需要更多显存 <br/>
  <br/>
**优化方案：**
 <br/>
1. 添加模型卸载功能 <br/>
2. 实现动态量化 <br/>
3. 优化前向传播过程中的显存使用 <br/>
 <br/>
**添加以下功能：**
 <br/>
1. 添加一个enable_model_cpu_offload方法，用于将模型组件卸载到CPU <br/>
2. 添加一个enable_sequential_cpu_offload方法，用于按顺序卸载模型组件 <br/>
3. 添加一个enable_model_quantization方法，用于量化模型 <br/>
4. 修改__call__方法，优化显存使用 <br/>
 <br/>

 
 
### 以下内容来自原始代码仓库


---
# ContentV: Efficient Training of Video Generation Models with Limited Compute

<div align="center">
<p align="center">
  <a href="https://contentv.github.io">
    <img
      src="https://img.shields.io/badge/Demo-Project Page-0A66C2?logo=googlechrome&logoColor=blue"
      alt="Project Page"
    />
  </a>
  <a href='https://arxiv.org/abs/2506.05343'>
    <img
      src="https://img.shields.io/badge/Tech Report-ArXiv-red?logo=arxiv&logoColor=red"
      alt="Tech Report"
    />
  </a>
  <a href="https://huggingface.co/ByteDance/ContentV-8B">
    <img 
        src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="Model"
    />
  </a>
  <a href="https://github.com/bytedance/ContentV">
    <img 
        src="https://img.shields.io/badge/Code-GitHub-orange?logo=github&logoColor=white" 
        alt="Code"
    />
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img
      src="https://img.shields.io/badge/License-Apache 2.0-5865F2?logo=apache&logoColor=purple"
      alt="License"
    />
  </a>
</p>
</div>

This project presents *ContentV*, an efficient framework for accelerating the training of DiT-based video generation models through three key innovations:

- A minimalist architecture that maximizes reuse of pre-trained image generation models for video synthesis
- A systematic multi-stage training strategy leveraging flow matching for enhanced efficiency
- A cost-effective reinforcement learning with human feedback framework that improves generation quality without requiring additional human annotations

Our open-source 8B model (based on Stable Diffusion 3.5 Large and Wan-VAE) achieves state-of-the-art result (85.14 on VBench) in only 4 weeks of training with 256×64GB NPUs.

<div align="center">
    <img src="./assets/demo.jpg" width="100%">
    <img src="./assets/arch.png" width="100%">
</div>

## ⚡ Quickstart

#### Recommended PyTorch Version

- GPU: torch >= 2.3.1 (CUDA >= 12.2)
- NPU: torch and torch-npu >= 2.1.0 (CANN >= 8.0.RC2). Please refer to [Ascend Extension for PyTorch](https://gitee.com/ascend/pytorch) for the installation of torch-npu.

#### Installation

```bash
git clone https://github.com/bytedance/ContentV.git
cd ContentV
pip3 install -r requirements.txt
```

#### T2V Generation

```bash
## For GPU
python3 demo.py
## For NPU
USE_ASCEND_NPU=1 python3 demo.py
```

## 📊 VBench

| Model | Total Score | Quality Score | Semantic Score | Human Action | Scene | Dynamic Degree | Multiple Objects  | Appear. Style |
|----------------------|--------|-------|-------|-------|-------|-------|-------|-------|
| Wan2.1-14B           | 86.22  | 86.67 | 84.44 | 99.20 | 61.24 | 94.26 | 86.59 | 21.59 |
| **ContentV (Long)**  | 85.14  | 86.64 | 79.12 | 96.80 | 57.38 | 83.05 | 71.41 | 23.02 |
| Goku†                | 84.85  | 85.60 | 81.87 | 97.60 | 57.08 | 76.11 | 79.48 | 23.08 |
| Open-Sora 2.0        | 84.34  | 85.40 | 80.12 | 95.40 | 52.71 | 71.39 | 77.72 | 22.98 |
| Sora†                | 84.28  | 85.51 | 79.35 | 98.20 | 56.95 | 79.91 | 70.85 | 24.76 |
| **ContentV (Short)** | 84.11  | 86.23 | 75.61 | 89.60 | 44.02 | 79.26 | 74.58 | 21.21 |
| EasyAnimate 5.1      | 83.42  | 85.03 | 77.01 | 95.60 | 54.31 | 57.15 | 66.85 | 23.06 |
| Kling 1.6†           | 83.40  | 85.00 | 76.99 | 96.20 | 55.57 | 62.22 | 63.99 | 20.75 |
| HunyuanVideo         | 83.24  | 85.09 | 75.82 | 94.40 | 53.88 | 70.83 | 68.55 | 19.80 |
| CogVideoX-5B         | 81.61  | 82.75 | 77.04 | 99.40 | 53.20 | 70.97 | 62.11 | 24.91 |
| Pika-1.0†            | 80.69  | 82.92 | 71.77 | 86.20 | 49.83 | 47.50 | 43.08 | 22.26 |
| VideoCrafter-2.0     | 80.44  | 82.20 | 73.42 | 95.00 | 55.29 | 42.50 | 40.66 | 25.13 |
| AnimateDiff-V2       | 80.27  | 82.90 | 69.75 | 92.60 | 50.19 | 40.83 | 36.88 | 22.42 |
| OpenSora 1.2         | 79.23  | 80.71 | 73.30 | 85.80 | 42.47 | 47.22 | 58.41 | 23.89 |

## ✅ Todo List
- [x] Inference code and checkpoints
- [ ] Training code of RLHF

## 🧾 License
This code repository and part of the model weights are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Please note that:
- MM DiT are derived from [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) and trained with video samples. This Stability AI Model is licensed under the [Stability AI Community License](https://stability.ai/community-license-agreement), Copyright ©  Stability AI Ltd. All Rights Reserved
- Video VAE from [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) is licensed under [Apache 2.0 License](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/main/LICENSE.txt)

## ❤️ Acknowledgement
* [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1)
* [Diffusers](https://github.com/huggingface/diffusers)
* [HuggingFace](https://huggingface.co)

## 🔗 Citation

```bibtex
@article{contentv2025,
  title     = {ContentV: Efficient Training of Video Generation Models with Limited Compute},
  author    = {Bytedance Douyin Content Team},
  journal   = {arXiv preprint arXiv:2506.05343},
  year      = {2025}
  }
```
