import os
import gradio as gr
import torch
import time
import uuid
from optimize_memory import load_optimized_pipeline, generate_video
from diffusers.utils import export_to_video

# 全局变量存储加载的pipeline
loaded_pipe = None

def initialize_model(
    model_path,
    device,
    enable_model_cpu_offload,
    progress=gr.Progress()
):
    """初始化模型"""
    global loaded_pipe
    
    progress(0.5, desc="正在加载模型...")
    try:
        loaded_pipe = load_optimized_pipeline(
            model_path=model_path,
            device=device,
            enable_model_cpu_offload=enable_model_cpu_offload
        )
        progress(1.0, desc="模型加载完成!")
        return "模型加载成功！"
    except Exception as e:
        return f"模型加载失败：{str(e)}"

def create_video(
    prompt,
    negative_prompt,
    num_frames,
    height,
    width,
    num_inference_steps,
    seed,
    progress=gr.Progress()
):
    """生成视频"""
    global loaded_pipe
    
    if loaded_pipe is None:
        return None, "请先加载模型！"
    
    try:
        progress(0.5, desc="正在生成视频...")
        
        # 确保outputs目录存在
        os.makedirs("outputs", exist_ok=True)
        
        # 生成唯一的文件名
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"outputs/video_{timestamp}_{unique_id}.mp4"
        
        # 生成视频
        video_frames = generate_video(
            pipeline=loaded_pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed if seed != -1 else None
        )
        
        progress(0.8, desc="正在保存视频...")
        
        # 保存视频
        export_to_video(video_frames[0], output_path, fps=24)
        
        progress(1.0, desc="视频生成完成!")
        return output_path, f"视频生成成功！已保存至: {output_path}"
    except Exception as e:
        return None, f"视频生成失败：{str(e)}"

def clear_gpu_memory():
    """清理GPU显存"""
    global loaded_pipe
    if loaded_pipe is not None:
        del loaded_pipe
        loaded_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "显存已清理！"
    return "没有加载的模型需要清理。"

# 创建Gradio界面
with gr.Blocks(title="ContentV Video Generation") as demo:
    gr.Markdown("# ContentV 视频生成")
    
    
    with gr.Tab("模型配置"):
        with gr.Row():
            with gr.Column():
                model_path = gr.Textbox(
                    label="模型路径",
                    value="ByteDance/ContentV-8B",
                    info="模型的本地路径或Hugging Face模型ID"
                )
                device = gr.Radio(
                    choices=["cuda", "cpu", "npu"],
                    value="cuda",
                    label="运行设备",
                    info="选择运行设备（推荐使用CUDA）"
                )
                
                # 显存优化选项
                with gr.Group():
                    gr.Markdown("### 显存优化选项")
                    enable_model_cpu_offload = gr.Checkbox(
                        label="启用模型CPU卸载",
                        value=True,
                        info="更激进的CPU卸载策略"
                    )
                
                load_button = gr.Button("加载模型")
                clear_button = gr.Button("清理显存")
                model_status = gr.Textbox(label="状态", interactive=False)
    
    with gr.Tab("视频生成"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="提示词",
                    value="A young musician sits on a rustic wooden stool in a cozy, dimly lit room, strumming an acoustic guitar with a worn, sunburst finish.",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    value="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
                    lines=2
                )
                
                with gr.Row():
                    num_frames = gr.Slider(
                        minimum=5,
                        maximum=125,
                        value=17,
                        step=1,
                        label="帧数"
                    )
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=30,
                        step=1,
                        label="推理步数"
                    )
                
                with gr.Row():
                    height = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=432,
                        step=8,
                        label="高度"
                    )
                    width = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=768,
                        step=8,
                        label="宽度"
                    )
                
                seed = gr.Slider(
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    value=42,
                    label="随机种子",
                    info="-1表示随机"
                )
                
                generate_button = gr.Button("生成视频")
                generation_status = gr.Textbox(label="状态", interactive=False)
            
            with gr.Column():
                video_output = gr.Video(label="生成的视频")
                
    # 显示使用说明
    gr.Markdown("""
    ## 使用说明
    
    1. **模型路径**: 输入已下载的ContentV模型路径
    2. **提示词设置**:
       - 正面提示词：描述您想要生成的视频内容
       - 负面提示词：描述您不想在视频中出现的内容
    3. **视频参数**:
       - 总帧数：视频的总帧数（影响生成时间）
       - 帧率：视频的播放速度
       - 分辨率：视频的宽度和高度
       - 推理步数：生成质量（步数越多质量越好，但耗时更长）
    4. **内存优化选项**:
       - 顺序CPU卸载：将不活跃的模型组件逐个卸载到CPU，适用于显存较小的GPU
       - 模型CPU卸载：整个模型级别的CPU卸载，比顺序卸载性能更好
       - VAE切片：将VAE输入分片处理，减少显存使用
       - VAE平铺：将VAE输入平铺处理，可以处理更大的图像
       - 注意力切片：将注意力计算分片处理，减少显存使用但可能略微降低速度
    
    注意：
    1. 顺序CPU卸载和模型CPU卸载只能选择其中一个
    2. 其他优化选项可以组合使用
    3. 生成视频可能需要一些时间，请耐心等待
    4. 生成的视频将保存在outputs目录下
    """)
    
    # 事件处理
    load_button.click(
        fn=initialize_model,
        inputs=[
            model_path,
            device,
            enable_model_cpu_offload
        ],
        outputs=model_status
    )
    
    clear_button.click(
        fn=clear_gpu_memory,
        inputs=[],
        outputs=model_status
    )
    
    generate_button.click(
        fn=create_video,
        inputs=[
            prompt,
            negative_prompt,
            num_frames,
            height,
            width,
            num_inference_steps,
            seed
        ],
        outputs=[video_output, generation_status]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()