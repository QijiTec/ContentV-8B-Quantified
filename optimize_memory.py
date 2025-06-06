import os
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from contentv_transformer import SD3Transformer3DModel
from contentv_pipeline import ContentVPipeline

USE_ASCEND_NPU = int(os.getenv('USE_ASCEND_NPU', '0'))
if USE_ASCEND_NPU:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False

def load_optimized_pipeline(
    model_path="ByteDance/ContentV-8B",  # 使用默认的ByteDance/ContentV-8B模型
    device="cuda",
    enable_cpu_offload=True,
    enable_model_cpu_offload=False
):
    """
    加载优化后的ContentV pipeline
    
    Args:
        model_path: 模型路径，默认为ByteDance/ContentV-8B
        device: 运行设备 ('cuda', 'cpu', 'npu')
        enable_cpu_offload: 是否启用顺序CPU卸载
        enable_model_cpu_offload: 是否启用模型CPU卸载
    """
    # 清理现有显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载VAE模型，使用float32
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    
    # 加载Transformer模型，使用bfloat16
    transformer = SD3Transformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # 创建pipeline，使用bfloat16
    pipe = ContentVPipeline.from_pretrained(
        model_path,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    # 移动到指定设备
    if device == "npu":
        pipe = pipe.to(device)
        if hasattr(pipe, 'generator'):
            pipe.generator = transfer_to_npu(pipe.generator)
    else:
        pipe = pipe.to(device)

    # 应用CPU卸载优化
    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    elif enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    return pipe

def generate_video(
    pipeline,
    prompt,
    negative_prompt,
    num_frames=16,
    height=432,
    width=768,
    num_inference_steps=30,
    seed=None,
    **kwargs
):
    """
    使用优化后的pipeline生成视频
    
    Args:
        pipeline: ContentV pipeline实例
        prompt: 正面提示词
        negative_prompt: 负面提示词
        num_frames: 视频帧数
        height: 视频高度
        width: 视频宽度
        num_inference_steps: 推理步数
        seed: 随机种子（可选）
    """
    # 设置随机种子
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None

    # 生成视频
    video = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **kwargs
    ).frames

    return video