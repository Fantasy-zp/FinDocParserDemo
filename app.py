"""
FinDocParser - Phase 3.4 版本
优化展示：美化界面 + 增强交互 + 中文化
"""
import gradio as gr
from pathlib import Path
import config
import utils
import base64
from io import BytesIO


def split_markdown_and_raw(combined_markdown):
    """
    分离处理后的 markdown 和原始输出
    
    Args:
        combined_markdown: 可能包含隐藏原始内容的 markdown
    
    Returns:
        (markdown_for_preview, raw_for_source)
    """
    # 检查是否为空
    if not combined_markdown or not isinstance(combined_markdown, str):
        return "", ""
    
    # 检查是否包含隐藏的原始内容
    if "<!-- RAW_OUTPUT_START" in combined_markdown and "RAW_OUTPUT_END -->" in combined_markdown:
        try:
            # 分离干净的 markdown
            parts = combined_markdown.split("<!-- RAW_OUTPUT_START")
            markdown_clean = parts[0].strip()
            
            # 提取原始内容
            raw_section = parts[1].split("RAW_OUTPUT_END -->")[0]
            raw_content = raw_section.strip()
            print(f"✅ 成功分离 - Markdown: {len(markdown_clean)} 字符, Raw: {len(raw_content)} 字符")

            # ✅ 返回两个值
            return markdown_clean, raw_content
        except Exception as e:
            print(f"⚠️  分离内容时出错: {e}")
            return combined_markdown, combined_markdown
    
    # 没有隐藏内容，Preview 和 Source 显示相同
    return combined_markdown, combined_markdown


# 在文件开头添加函数
def get_logo_html():
    """生成 Logo HTML"""
    logo_path = "assets/logo.png"
    
    try:
        with open(logo_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        
        return f"""
        <div style='
            display: flex;
            flex-direction: row; /* 水平布局 */
            align-items: center;
            justify-content: flex-start;
            padding: 0;
            margin-top: 0px; /* 减少顶部间距 */
        '>
            <img src='data:image/jpeg;base64,{img_data}' 
                 alt='建设银行' 
                 style='
                     width: 30px; 
                     height: 30px; 
                     object-fit: contain;
                     margin-right: 10px; /* 图片右侧间距 */
                 '/>
            <p style='
                font-size: 15px; 
                color: #555; 
                margin: 0;
                font-weight: 500;
                line-height: 1.3;
                text-align: left; /* 文字左对齐 */
            '>
                <strong>集团金融科技创新中心</strong>
            </p>
        </div>
        """
    except Exception as e:
        print(f"⚠️  加载 Logo 失败: {e}")
        return """
        <div style='display: flex; flex-direction: row; align-items: center; padding: 0; margin-top: -15px;'>
            <p style='font-size: 45px; margin: 0 10px 0 0;'>🏦</p>
            <p style='font-size: 11px; color: #555; margin: 0; line-height: 1.3; text-align: left;'>
                集团金融科技创新中心
            </p>
        </div>
        """

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
    """创建下载按钮的 HTML（只下载干净的 markdown）"""
    if not markdown:
        return None
    
    # ✅ 使用已有函数清理内容
    clean_markdown, _ = split_markdown_and_raw(markdown)
    
    # 创建可下载的文件
    b64 = base64.b64encode(clean_markdown.encode()).decode()
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
        📥 下载 Markdown
    </a>
    """


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
                         Roboto, "Helvetica Neue", Arial, "PingFang SC", 
                         "Microsoft YaHei", sans-serif !important;
        }
        
        /* 状态框样式 */
        #status-box {
            font-family: 'Courier New', 'Microsoft YaHei', monospace !important;
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
        
        /* Logo 容器样式 */
        #logo-container {
            margin-top: -5px; /* 进一步减少顶部间距 */
        }
        
        /* 响应式布局 */
        @media (max-width: 768px) {
            #original-gallery img {
                max-height: 400px;
            }
            
            /* 响应式：移动端 Logo 居中 */
            #logo-container {
                margin-top: 0;
            }
        }
    """
) as demo:

    # 标题和 Logo
    with gr.Row(equal_height=False):
        # 左侧：标题和描述
        with gr.Column(scale=8):
            gr.Markdown(f"# {config.TITLE}")
            gr.Markdown(config.DESCRIPTION)
        
        # 右侧：Logo 和说明
        with gr.Column(scale=2, min_width=100, elem_id="logo-container"):
            gr.HTML(get_logo_html())
    
    with gr.Row():
        # ============================================
        # 左侧：输入面板
        # ============================================
        with gr.Column(scale=4):
            gr.Markdown("### 📄 上传文档")
            
            file_input = gr.File(
                label="上传 PDF 或图片",
                file_types=config.ALLOWED_FILE_TYPES,
                type="filepath"
            )
            
            # 模型选择
            gr.Markdown("### 🤖 模型选择")
            model_dropdown = gr.Dropdown(
                choices=utils.get_model_choices(),
                value=config.MODELS[config.DEFAULT_MODEL]["name"],
                label="选择模型",
                info="选择用于解析的模型"
            )
            
            # 测试模型按钮
            with gr.Row():
                test_btn = gr.Button("🔍 测试连接", scale=1)
                test_result = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    scale=2,
                    placeholder="点击测试模型连接"
                )
            # test_btn = gr.Button("🔍 测试连接", size="lg", variant="secondary")
            # test_result = gr.Textbox(
            #     show_label=False,
            #     interactive=False,
            #     placeholder="点击测试模型连接",
            #     lines=1
            # )

            
            # 高级设置
            with gr.Accordion("⚙️ 高级设置", open=False):
                temperature = gr.Slider(
                    0.0, 1.0, 
                    value=config.DEFAULT_TEMPERATURE,
                    step=0.0001,
                    label="温度参数",
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
                    label="最大生成长度",
                    info="最大 token 数量"
                )
                
                custom_prompt = gr.Textbox(
                    label="自定义提示词（可选）",
                    placeholder="留空使用默认提示词",
                    lines=4
                )
            
            # 缓存管理
            with gr.Accordion("💾 缓存管理", open=False):
                with gr.Row():
                    cache_stats_btn = gr.Button("📊 查看统计", size="sm")
                    clear_cache_btn = gr.Button("🧹 清空缓存", size="sm", variant="stop")
                
                cache_info = gr.Textbox(
                    label="缓存信息",
                    interactive=False,
                    lines=8
                )
            
            # 解析按钮
            parse_btn = gr.Button(
                "🚀 开始解析",
                variant="primary",
                size="lg",
                elem_classes="primary-button"
            )
            
            # 状态显示
            status_box = gr.Textbox(
                label="状态",
                interactive=False,
                placeholder="准备就绪，等待解析...",
                lines=8,
                elem_id="status-box"
            )
        
        # ============================================
        # 右侧：输出面板
        # ============================================
        with gr.Column(scale=6):
            gr.Markdown("### 📊 解析结果")
            
            # 下载按钮（在标签页外）
            download_html = gr.HTML()
            
            with gr.Tabs():
                with gr.Tab("📄 原始文档"):
                    original_gallery = gr.Gallery(
                        label="文档页面",
                        columns=2,
                        rows=None,
                        height=None,
                        object_fit="contain",
                        show_label=False,
                        elem_id="original-gallery",
                        allow_preview=True,
                        preview=True
                    )
                
                with gr.Tab("👁️ 预览"):
                    markdown_preview = gr.Markdown(
                        value="",
                        elem_id="markdown-preview"
                    )
                
                with gr.Tab("</> 源码"):
                    markdown_source = gr.Code(
                        value="",
                        language="markdown",
                        lines=30,
                        elem_id="markdown-source"
                    )
    
    # ============================================
    # Examples
    # ============================================
    gr.Markdown("---")
    gr.Markdown("### 📚 示例文档")
    
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
                        zoom = 2
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        doc.close()
                        
                        max_size = 600
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        preview_images.append((img, f.stem))
                    else:
                        img = Image.open(f)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        max_size = 600
                        img.thumbnail((max_size, max_size), Image.LANCZOS)
                        preview_images.append((img, f.stem))
                        
                except Exception as e:
                    print(f"⚠️  加载示例失败 {f.name}: {e}")
                    continue
            
            if preview_images:
                gr.Markdown("*点击示例文档即可加载*")
                
                example_gallery = gr.Gallery(
                    value=preview_images,
                    label=None,
                    show_label=False,
                    columns=3,
                    rows=None,
                    height=None,
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
                gr.Markdown("*加载示例预览失败*")
        else:
            gr.Markdown(f"*在 `{config.EXAMPLES_DIR}` 目录中未找到示例文件*")
    else:
        gr.Markdown(f"*示例目录不存在：`{config.EXAMPLES_DIR}`*")
    
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
    # markdown_preview.change(
    #     fn=lambda x: x,
    #     inputs=[markdown_preview],
    #     outputs=[markdown_source]
    # )
    # 同步预览和源码（分离原始内容）
    markdown_preview.change(
        fn=lambda x: split_markdown_and_raw(x)[1],  # 只取原始内容
        inputs=[markdown_preview],
        outputs=[markdown_source]  # 只更新 Source
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
        💡 **使用提示：** 
        - ✅ 支持 多页PDF 和图片文件
        - ✅ 实时显示处理进度和结果
        - ✅ 支持下载 Markdown 文件
        - ✅ 智能缓存，重复文档秒返回
        """
    )


if __name__ == "__main__":
    print("🚀 启动 FinDocParser Demo - Phase 3.4...")
    print(f"🌐 访问地址: http://localhost:7860")
    print(f"📋 可用模型数量: {len(config.MODELS)}")
    for key, model in config.MODELS.items():
        print(f"  - {model['name']}: {model['api_base']}")
    print("\n" + "="*80)
    print("✨ Phase 3.4 功能:")
    print("  - 📄 Gallery 无限滚动（支持任意页数）")
    print("  - 📥 Markdown 下载功能")
    print("  - 🎨 美化界面样式")
    print("  - 📱 响应式布局")
    print("  - ⚡ 优化加载体验")
    print("  - 🇨🇳 完整中文界面")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )