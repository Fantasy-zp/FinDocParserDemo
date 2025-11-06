"""
FinDocParser - Phase 3.4 版本
优化展示：美化界面 + 增强交互
"""
import gradio as gr
from pathlib import Path
import config
import utils
import base64
from io import BytesIO


def parse_document_streaming(
    file, 
    model_name,
    temperature,
    top_p,
    max_tokens,
    custom_prompt
):
    """流式解析文档（Phase 3.4 优化版）"""
    try:
        # 验证文件
        is_valid, error_msg = utils.validate_file(file)
        if not is_valid:
            yield None, f"❌ 错误：{error_msg}", "", None
            return
        
        # 获取模型键
        model_key = utils.get_model_key_from_name(model_name)
        
        # 使用自定义 Prompt 或默认 Prompt
        prompt = custom_prompt.strip() if custom_prompt.strip() else config.DEFAULT_PROMPT
        
        # 流式处理
        for images, status, markdown, from_cache in utils.process_document_streaming_with_cache(
            file, 
            model_key, 
            prompt,
            temperature,
            top_p,
            max_tokens
        ):
            # 添加缓存标识
            if from_cache:
                status = "⚡ " + status
            
            # 生成下载链接
            download_btn = create_download_button(markdown, Path(file).stem)
            
            yield images, status, markdown, download_btn
        
    except Exception as e:
        error_msg = f"""❌ 解析失败

错误信息：{str(e)}

请检查：
1. 文件格式是否正确
2. 模型服务是否正常运行
3. 网络连接是否正常

如问题持续，请联系技术支持。"""
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield None, error_msg, "", None


def create_download_button(markdown, filename):
    """创建下载按钮的 HTML"""
    if not markdown:
        return None
    
    # 创建可下载的文件
    b64 = base64.b64encode(markdown.encode()).decode()
    href = f'data:text/markdown;base64,{b64}'
    
    return f"""
    <a href="{href}" download="{filename}.md" style="
        display: inline-block;
        padding: 8px 16px;
        background: #3b82f6;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
        transition: background 0.2s;
    " onmouseover="this.style.background='#2563eb'" 
       onmouseout="this.style.background='#3b82f6'">
        📥 Download Markdown
    </a>
    """


def copy_to_clipboard(text):
    """复制到剪贴板的反馈"""
    if text:
        return "✅ Copied to clipboard!"
    return "⚠️ Nothing to copy"


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
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid #3b82f6;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Gallery 优化 */
        #original-gallery {
            min-height: 600px;
            max-height: 800px;
            overflow-y: auto !important;
        }
        
        #original-gallery img {
            object-fit: contain !important;
            width: 100% !important;
            height: auto !important;
            max-height: 1000px;
        }
        
        /* Source 代码框滚动 */
        #markdown-source {
            max-height: 600px;
            overflow-y: auto !important;
        }
        
        /* Preview 滚动 */
        #markdown-preview {
            max-height: 600px;
            overflow-y: auto !important;
            padding: 16px;
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
        
        #example-gallery::-webkit-scrollbar,
        #original-gallery::-webkit-scrollbar,
        #markdown-source::-webkit-scrollbar,
        #markdown-preview::-webkit-scrollbar {
            width: 8px;
        }
        
        #example-gallery::-webkit-scrollbar-track,
        #original-gallery::-webkit-scrollbar-track,
        #markdown-source::-webkit-scrollbar-track,
        #markdown-preview::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        #example-gallery::-webkit-scrollbar-thumb,
        #original-gallery::-webkit-scrollbar-thumb,
        #markdown-source::-webkit-scrollbar-thumb,
        #markdown-preview::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        #example-gallery::-webkit-scrollbar-thumb:hover,
        #original-gallery::-webkit-scrollbar-thumb:hover,
        #markdown-source::-webkit-scrollbar-thumb:hover,
        #markdown-preview::-webkit-scrollbar-thumb:hover {
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
        
        /* 按钮动画 */
        .primary-button {
            transition: all 0.3s ease;
        }
        
        .primary-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* 加载动画 */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading {
            animation: spin 1s linear infinite;
        }
        
        /* 响应式布局 */
        @media (max-width: 768px) {
            #original-gallery img {
                max-height: 400px;
            }
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
            
            # 缓存管理
            with gr.Accordion("💾 Cache Management", open=False):
                with gr.Row():
                    cache_stats_btn = gr.Button("📊 View Stats", size="sm")
                    clear_cache_btn = gr.Button("🧹 Clear Cache", size="sm", variant="stop")
                
                cache_info = gr.Textbox(
                    label="Cache Information",
                    interactive=False,
                    lines=8
                )
            
            # 解析按钮
            parse_btn = gr.Button(
                "🚀 Parse Document",
                variant="primary",
                size="lg",
                elem_classes="primary-button"
            )
            
            # 状态显示
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Ready to parse...",
                lines=8,
                elem_id="status-box"
            )
        
        # ============================================
        # 右侧：输出面板
        # ============================================
        with gr.Column(scale=6):
            gr.Markdown("### 📊 Results")
            
            # 下载按钮（在标签页外）
            download_html = gr.HTML()
            
            with gr.Tabs():
                with gr.Tab("📄 Original"):
                    original_gallery = gr.Gallery(
                        label="Document Pages",
                        columns=2,
                        rows=None,  # ✅ 移除行数限制
                        height=None,  # ✅ 自动高度
                        object_fit="contain",
                        show_label=False,
                        elem_id="original-gallery",
                        allow_preview=True,
                        preview=True
                    )
                
                with gr.Tab("👁️ Preview"):
                    with gr.Row():
                        copy_preview_btn = gr.Button("📋 Copy", size="sm")
                    
                    markdown_preview = gr.Markdown(
                        value="",
                        elem_id="markdown-preview"
                    )
                
                with gr.Tab("</> Source"):
                    with gr.Row():
                        copy_source_btn = gr.Button("📋 Copy", size="sm")
                    
                    markdown_source = gr.Code(
                        value="",
                        language="markdown",
                        lines=30,  # ✅ 增加行数
                        elem_id="markdown-source"
                    )
            
            # 复制反馈
            copy_feedback = gr.Textbox(
                show_label=False,
                interactive=False,
                visible=False
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
                    rows=None,  # ✅ 移除行数限制
                    height=None,  # ✅ 自动高度
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
    
    # 流式解析文档
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
        outputs=[original_gallery, status_box, markdown_preview, download_html]
    )
    
    # 同步预览和源码
    markdown_preview.change(
        fn=lambda x: x,
        inputs=[markdown_preview],
        outputs=[markdown_source]
    )
    
    # 复制功能
    copy_preview_btn.click(
        fn=copy_to_clipboard,
        inputs=[markdown_preview],
        outputs=[copy_feedback]
    )
    
    copy_source_btn.click(
        fn=copy_to_clipboard,
        inputs=[markdown_source],
        outputs=[copy_feedback]
    )
    
    # 缓存管理
    cache_stats_btn.click(
        fn=utils.get_cache_stats,
        outputs=[cache_info]
    )

    clear_cache_btn.click(
        fn=utils.clear_cache,
        outputs=[cache_info]
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
        - ✅ 支持下载 Markdown 文件
        - ✅ 支持复制到剪贴板
        - ✅ 智能缓存，重复文档秒返回
        - 🚀 Phase 3.4: 优化展示 + 增强交互
        """
    )


if __name__ == "__main__":
    print("🚀 Starting FinDocParser Demo - Phase 3.4...")
    print(f"🌐 Interface: http://localhost:7860")
    print(f"📋 Available models: {len(config.MODELS)}")
    for key, model in config.MODELS.items():
        print(f"  - {model['name']}: {model['api_base']}")
    print("\n" + "="*80)
    print("✨ Phase 3.4 Features:")
    print("  - 📄 Gallery 无限滚动（支持任意页数）")
    print("  - 📥 Markdown 下载功能")
    print("  - 📋 一键复制到剪贴板")
    print("  - 🎨 美化界面样式")
    print("  - 📱 响应式布局")
    print("  - ⚡ 优化加载体验")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )