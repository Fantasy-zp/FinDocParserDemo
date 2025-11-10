# test_parallel.py
import time
from pathlib import Path
import config
import utils

# 模拟测试
test_pdf = "examples/1223836236.pdf"  # 替换为你的测试文件

print("="*60)
print("并行处理性能测试")
print("="*60)

# 测试 1: 禁用并行
print("\n📊 测试 1: 串行处理")
config.PARALLEL_ENABLED = False
start = time.time()
images, markdown = utils.process_document(
    test_pdf,
    "qwen2_5vl_finetuned",
    config.DEFAULT_PROMPT,
    config.DEFAULT_TEMPERATURE,
    config.DEFAULT_TOP_P,
    config.DEFAULT_MAX_TOKENS
)
serial_time = time.time() - start
print(f"⏱️  耗时: {serial_time:.2f}s")

# 测试 2: 启用并行
print("\n📊 测试 2: 并行处理")
config.PARALLEL_ENABLED = True
start = time.time()
images, markdown = utils.process_document(
    test_pdf,
    "qwen2_5vl_finetuned",
    config.DEFAULT_PROMPT,
    config.DEFAULT_TEMPERATURE,
    config.DEFAULT_TOP_P,
    config.DEFAULT_MAX_TOKENS
)
parallel_time = time.time() - start
print(f"⏱️  耗时: {parallel_time:.2f}s")

# 对比
print("\n" + "="*60)
print(f"提速比: {serial_time / parallel_time:.2f}x")
print("="*60)