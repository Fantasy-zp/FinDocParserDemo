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
        
        /* ✅ Examples Gallery 优化 */
        #example-gallery {
            max-height: 600px;
            overflow-y: auto;  /* 允许垂直滚动 */
            overflow-x: hidden;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px;
            background: #fafafa;
        }
        
        /* 滚动条美化 */
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
        
        /* Gallery 项样式 */
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
    # Examples（优化版 - 清晰 + 可滚动）
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
            max_examples = min(12, len(example_files))  # 最多 12 个
            
            for f in example_files[:max_examples]:
                try:
                    if f.suffix.lower() == '.pdf':
                        # ✅ 提高 PDF 缩略图分辨率
                        doc = fitz.open(str(f))
                        page = doc[0]
                        # 使用固定 DPI 生成高质量缩略图
                        zoom = 2.0  # 提高缩放比例（原来是隐式的低分辨率）
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        doc.close()
                        
                        # 调整到合适大小（保持清晰度）
                        max_size = 600  # 提高到 600px
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        preview_images.append((img, f.stem))
                    else:
                        # ✅ 图片使用高质量缩放
                        img = Image.open(f)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        # 高质量缩放
                        max_size = 400
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        preview_images.append((img, f.stem))
                        
                except Exception as e:
                    print(f"⚠️  Failed to load example {f.name}: {e}")
                    continue
            
            if preview_images:
                gr.Markdown("*Click an example to load it*")
                
                # ✅ 改进 Gallery 配置
                example_gallery = gr.Gallery(
                    value=preview_images,
                    label=None,
                    show_label=False,
                    columns=4,
                    rows=2,  # 增加行数
                    height="auto",  # ✅ 改为 auto，允许自适应
                    object_fit="scale-down",  # ✅ 改为 scale-down 保持清晰度
                    allow_preview=True,  # ✅ 允许预览大图
                    show_download_button=False,
                    container=True,  # ✅ 添加容器
                    elem_id="example-gallery"
                )
                
                # 点击事件
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