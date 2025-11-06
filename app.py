"""
FinDocParser - Phase 3.2 版本
功能增强：并行处理 + 流式输出 + 实时进度
"""
import gradio as gr
from pathlib import Path
import config
import utils


def parse_document_streaming(
    file, 
    model_name,
    temperature,
    top_p,
    max_tokens,
    custom_prompt
):
    """
    流式解析文档（Phase 3.2）
    
    每完成一页就返回当前状态
    """
    try:
        # 验证文件
        is_valid, error_msg = utils.validate_file(file)
        if not is_valid:
            yield None, f"❌ 错误：{error_msg}", ""
            return
        
        # 获取模型键
        model_key = utils.get_model_key_from_name(model_name)
        
        # 使用自定义 Prompt 或默认 Prompt
        prompt = custom_prompt.strip() if custom_prompt.strip() else config.DEFAULT_PROMPT
        
        # ✅ 流式处理
        for images, status, markdown in utils.process_document_streaming(
            file, 
            model_key, 
            prompt,
            temperature,
            top_p,
            max_tokens
        ):
            yield images, status, markdown
        
    except Exception as e:
        error_msg = f"❌ 解析失败：{str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield None, error_msg, ""


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
        
        /* 状态框样式 */
        #status-box {
            font-family: 'Courier New', monospace !important;
            background: #f8f9fa;
            border-left: 4px solid #3b82f6;
            padding: 12px;
        }
        
        /* Examples Gallery 样式 */
        #example-gallery {
            max-height: 600px;
            overflow-y: auto;
            overflow-x: hidden;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px;
            background: #fafafa;
        }
        
        #example-gallery::-webkit-scrollbar {
            width: 8px;
        }
        
        #example-gallery::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        #example-gallery::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        #example-gallery::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        #example-gallery img {
            border: 2px solid transparent;
            border-radius: 6px;
            transition: all 0.2s ease;
            cursor: pointer;
            background: white;
            padding: 4px;
        }
        
        #example-gallery img:hover {
            border-color: #3b82f6;
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
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
            
            # 测试模型按钮
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
                    step=0.0001,
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
            
            # ✅ 状态显示（支持实时更新）
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Ready to parse...",
                lines=6,
                elem_id="status-box"
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
    
    examples_dir = Path(config.EXAMPLES_DIR)
    
    if examples_dir.exists() and examples_dir.is_dir():
        example_files = sorted(
            list(examples_dir.glob("*.png")) + 
            list(examples_dir.glob("*.jpg")) + 
            list(examples_dir.glob("*.jpeg")) + 
            list(examples_dir.glob("*.pdf"))
        )
        
        if example_files:
            from PIL import Image
            import fitz
            
            preview_images = []
            max_examples = min(12, len(example_files))
            
            for f in example_files[:max_examples]:
                try:
                    if f.suffix.lower() == '.pdf':
                        doc = fitz.open(str(f))
                        page = doc[0]
                        zoom = 1.5
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        doc.close()
                        
                        max_size = 400
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        preview_images.append((img, f.stem))
                    else:
                        img = Image.open(f)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        max_size = 400
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        preview_images.append((img, f.stem))
                        
                except Exception as e:
                    print(f"⚠️  Failed to load example {f.name}: {e}")
                    continue
            
            if preview_images:
                gr.Markdown("*Click an example to load it*")
                
                example_gallery = gr.Gallery(
                    value=preview_images,
                    label=None,
                    show_label=False,
                    columns=4,
                    rows=3,
                    height="auto",
                    object_fit="scale-down",
                    allow_preview=True,
                    show_download_button=False,
                    container=True,
                    elem_id="example-gallery"
                )
                
                def load_example_file(evt: gr.SelectData):
                    idx = evt.index
                    if 0 <= idx < len(example_files):
                        return str(example_files[idx]), config.MODELS[config.DEFAULT_MODEL]["name"]
                    return None, None
                
                example_gallery.select(
                    fn=load_example_file,
                    outputs=[file_input, model_dropdown]
                )
            else:
                gr.Markdown("*Failed to load example previews*")
        else:
            gr.Markdown("*No example files found in `examples/` directory*")
    else:
        gr.Markdown(f"*Examples directory not found: `{config.EXAMPLES_DIR}`*")
    
    # ============================================
    # 事件绑定
    # ============================================
    
    # 测试模型连接
    test_btn.click(
        fn=test_model,
        inputs=[model_dropdown],
        outputs=[test_result]
    )
    
    # ✅ 流式解析文档
    parse_btn.click(
        fn=parse_document_streaming,
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
        - ✅ 支持 PDF 和图片文件
        - ✅ 实时显示处理进度和结果
        - ✅ 可切换不同模型对比效果
        - ✅ 高级设置中可自定义参数
        - 🚀 Phase 3.2: 流式输出 + 实时进度条
        """
    )


if __name__ == "__main__":
    print("🚀 Starting FinDocParser Demo - Phase 3.2...")
    print(f"🌐 Interface: http://localhost:7860")
    print(f"📋 Available models: {len(config.MODELS)}")
    for key, model in config.MODELS.items():
        print(f"  - {model['name']}: {model['api_base']}")
    print("\n" + "="*80)
    print("✨ New Features:")
    print("  - Real-time streaming output")
    print("  - Live progress tracking")
    print("  - Instant result preview")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )