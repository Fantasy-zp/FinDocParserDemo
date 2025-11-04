"""
FinDocParser - MVP 版本
"""
import gradio as gr
from pathlib import Path
import config
import utils
import traceback


def parse_document(file, progress=gr.Progress()):
    """解析文档"""
    try:
        print("\n" + "="*80)
        print("🚀 NEW REQUEST")
        print("="*80)
        
        # 验证文件
        is_valid, error_msg = utils.validate_file(file)
        if not is_valid:
            print(f"❌ Validation failed: {error_msg}")
            return None, f"❌ 错误：{error_msg}", ""
        
        print(f"✅ File validated: {file}")
        
        # 处理文档
        progress(0, desc="Starting...")
        images, markdown = utils.process_document(file, progress)
        
        success_msg = f"✅ 解析完成！共 {len(images)} 页，生成 {len(markdown)} 个字符"
        print(f"\n{success_msg}")
        
        return images, success_msg, markdown
        
    except Exception as e:
        error_detail = f"❌ 解析失败：{type(e).__name__}: {str(e)}"
        print(f"\n{error_detail}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None, error_detail, ""


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
    
    gr.Markdown(f"# {config.TITLE}")
    gr.Markdown(config.DESCRIPTION)
    gr.Markdown(f"**Current Model:** {config.MODEL_NAME}")
    
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 📄 Upload Document")
            
            file_input = gr.File(
                label="Upload PDF or Image",
                file_types=config.ALLOWED_FILE_TYPES,
                type="filepath"
            )
            
            parse_btn = gr.Button(
                "🚀 Parse Document",
                variant="primary",
                size="lg"
            )
            
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Ready to parse...",
                lines=3
            )
        
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
    
    parse_btn.click(
        fn=parse_document,
        inputs=[file_input],
        outputs=[original_gallery, status_box, markdown_preview]
    )
    
    markdown_preview.change(
        fn=lambda x: x,
        inputs=[markdown_preview],
        outputs=[markdown_source]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        "💡 **Tips:** 支持 PDF 或图片。确保模型服务已启动（端口 8001）。"
    )


if __name__ == "__main__":
    print("🚀 Starting FinDocParser Demo...")
    print(f"📡 Model API: {config.MODEL_API_BASE}")
    print(f"🌐 Interface will be available at: http://localhost:7860")
    print("\n" + "="*80)
    print("Waiting for requests...")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )