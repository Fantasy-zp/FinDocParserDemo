"""
FinDocParser - Phase 2 版本
功能增强：多模型 + 高级设置 + Examples
"""
import gradio as gr
from pathlib import Path
import config
import utils


def parse_document(
    file, 
    model_name,
    temperature,
    top_p,
    max_tokens,
    custom_prompt,
    progress=gr.Progress()
):
    """
    解析文档（支持多模型和自定义参数）
    """
    try:
        # 验证文件
        is_valid, error_msg = utils.validate_file(file)
        if not is_valid:
            return None, f"❌ 错误：{error_msg}", ""
        
        # 获取模型键
        model_key = utils.get_model_key_from_name(model_name)
        
        # 使用自定义 Prompt 或默认 Prompt
        prompt = custom_prompt.strip() if custom_prompt.strip() else config.DEFAULT_PROMPT
        
        # 处理文档
        progress(0, desc="Starting...")
        images, markdown = utils.process_document(
            file, 
            model_key, 
            prompt,
            temperature,
            top_p,
            max_tokens,
            progress
        )
        
        success_msg = f"✅ 解析完成！共 {len(images)} 页，使用模型：{model_name}"
        return images, success_msg, markdown
        
    except Exception as e:
        error_msg = f"❌ 解析失败：{str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg, ""


def test_model(model_name):
    """测试模型连接"""
    model_key = utils.get_model_key_from_name(model_name)
    success, message = utils.test_model_connection(model_key)
    return message


# ============================================
# Gradio 界面
# ============================================
with gr.Blocks(
    title=config.TITLE,
    theme=gr.themes.Soft(),
    css="""
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", 
                         Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
    """
) as demo:
    
    # 标题
    gr.Markdown(f"# {config.TITLE}")
    gr.Markdown(config.DESCRIPTION)
    
    with gr.Row():
        # ============================================
        # 左侧：输入面板
        # ============================================
        with gr.Column(scale=4):
            gr.Markdown("### 📄 Upload Document")
            
            file_input = gr.File(
                label="Upload PDF or Image",
                file_types=config.ALLOWED_FILE_TYPES,
                type="filepath"
            )
            
            # 模型选择
            gr.Markdown("### 🤖 Model Selection")
            model_dropdown = gr.Dropdown(
                choices=utils.get_model_choices(),
                value=config.MODELS[config.DEFAULT_MODEL]["name"],
                label="Select Model",
                info="选择用于解析的模型"
            )
            
            # 测试模型按钮（小按钮）
            with gr.Row():
                test_btn = gr.Button("🔍 Test Connection", size="sm", scale=1)
                test_result = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    scale=2,
                    placeholder="Click to test model connection"
                )
            
            # 高级设置
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                temperature = gr.Slider(
                    0.0, 1.0, 
                    value=config.DEFAULT_TEMPERATURE,
                    step=0.1,
                    label="Temperature",
                    info="较低值更确定，较高值更随机"
                )
                
                top_p = gr.Slider(
                    0.0, 1.0,
                    value=config.DEFAULT_TOP_P,
                    step=0.1,
                    label="Top P",
                    info="核采样参数"
                )
                
                max_tokens = gr.Slider(
                    1024, 16384,
                    value=config.DEFAULT_MAX_TOKENS,
                    step=1024,
                    label="Max Tokens",
                    info="最大生成长度"
                )
                
                custom_prompt = gr.Textbox(
                    label="Custom Prompt (Optional)",
                    placeholder="留空使用默认 Prompt",
                    lines=4
                )
            
            # 解析按钮
            parse_btn = gr.Button(
                "🚀 Parse Document",
                variant="primary",
                size="lg"
            )
            
            # 状态显示
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Ready to parse...",
                lines=2
            )
        
        # ============================================
        # 右侧：输出面板
        # ============================================
        with gr.Column(scale=6):
            gr.Markdown("### 📊 Results")
            
            with gr.Tabs():
                with gr.Tab("📄 Original"):
                    original_gallery = gr.Gallery(
                        label="Document Pages",
                        columns=2,
                        height=600,
                        object_fit="contain",
                        show_label=False
                    )
                
                with gr.Tab("👁️ Preview"):
                    markdown_preview = gr.Markdown(
                        value="",
                        height=600
                    )
                
                with gr.Tab("</> Source"):
                    markdown_source = gr.Code(
                        value="",
                        language="markdown",
                        lines=20
                    )
    
    # ============================================
    # Examples
    # ============================================
    gr.Markdown("---")
    gr.Markdown("### 📚 Examples")
    
    # 如果有 examples 目录，显示示例
    examples_dir = Path(config.EXAMPLES_DIR)
    if examples_dir.exists():
        example_files = list(examples_dir.glob("*.png")) + list(examples_dir.glob("*.pdf"))
        if example_files:
            examples = gr.Examples(
                examples=[
                    [str(f), config.MODELS[config.DEFAULT_MODEL]["name"]]
                    for f in example_files[:4]  # 最多显示 4 个
                ],
                inputs=[file_input, model_dropdown],
                label="Click to load example"
            )
    
    # ============================================
    # 事件绑定
    # ============================================
    
    # 测试模型连接
    test_btn.click(
        fn=test_model,
        inputs=[model_dropdown],
        outputs=[test_result]
    )
    
    # 解析文档
    parse_btn.click(
        fn=parse_document,
        inputs=[
            file_input,
            model_dropdown,
            temperature,
            top_p,
            max_tokens,
            custom_prompt
        ],
        outputs=[original_gallery, status_box, markdown_preview]
    )
    
    # 同步预览和源码
    markdown_preview.change(
        fn=lambda x: x,
        inputs=[markdown_preview],
        outputs=[markdown_source]
    )
    
    # ============================================
    # 页脚
    # ============================================
    gr.Markdown("---")
    gr.Markdown(
        """
        💡 **Tips:** 
        - 支持 PDF 和图片文件
        - 可切换不同模型对比效果
        - 高级设置中可自定义参数
        - 确保对应的模型服务已启动
        """
    )


if __name__ == "__main__":
    print("🚀 Starting FinDocParser Demo - Phase 2...")
    print(f"🌐 Interface: http://localhost:7860")
    print(f"📋 Available models: {len(config.MODELS)}")
    for key, model in config.MODELS.items():
        print(f"  - {model['name']}: {model['api_base']}")
    print("\n" + "="*80)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )