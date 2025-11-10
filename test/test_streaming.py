# test_streaming.py
"""测试流式输出效果"""
import time
from pathlib import Path
import config
import utils

test_pdf = "examples/1223836236.pdf"  # 替换为你的测试文件

print("="*60)
print("流式输出测试")
print("="*60)

print("\n开始流式处理...\n")

for images, status, markdown in utils.process_document_streaming(
    test_pdf,
    "qwen2_5vl_finetuned",
    config.DEFAULT_PROMPT,
    config.DEFAULT_TEMPERATURE,
    config.DEFAULT_TOP_P,
    config.DEFAULT_MAX_TOKENS
):
    print(f"\n{'='*60}")
    print(status)
    print(f"{'='*60}")
    print(f"已生成 Markdown 字符数: {len(markdown)}")
    print(f"已处理图片数: {len(images)}")
    print()

print("\n✅ 流式处理完成！")