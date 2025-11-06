# test_cache.py
"""测试缓存功能"""
import time
from pathlib import Path
import config
import utils
from cache_manager import get_cache_manager

test_pdf = "examples/1223836236.pdf" 

print("="*60)
print("缓存功能测试")
print("="*60)

# 清空缓存
cache_mgr = get_cache_manager()
cache_mgr.clear_all()

# 第一次处理（无缓存）
print("\n📊 Test 1: 首次处理（无缓存）")
start = time.time()
images1, markdown1, from_cache1 = utils.process_document_with_cache(
    test_pdf,
    "qwen2_5vl_finetuned",
    config.DEFAULT_PROMPT,
    config.DEFAULT_TEMPERATURE,
    config.DEFAULT_TOP_P,
    config.DEFAULT_MAX_TOKENS
)
time1 = time.time() - start
print(f"⏱️  耗时: {time1:.2f}s")
print(f"💾 来自缓存: {from_cache1}")

# 第二次处理（应该命中缓存）
print("\n📊 Test 2: 重复处理（应该命中缓存）")
start = time.time()
images2, markdown2, from_cache2 = utils.process_document_with_cache(
    test_pdf,
    "qwen2_5vl_finetuned",
    config.DEFAULT_PROMPT,
    config.DEFAULT_TEMPERATURE,
    config.DEFAULT_TOP_P,
    config.DEFAULT_MAX_TOKENS
)
time2 = time.time() - start
print(f"⏱️  耗时: {time2:.2f}s")
print(f"💾 来自缓存: {from_cache2}")

# 第三次处理（修改参数，不应命中）
print("\n📊 Test 3: 修改参数（不应命中缓存）")
start = time.time()
images3, markdown3, from_cache3 = utils.process_document_with_cache(
    test_pdf,
    "qwen2_5vl_finetuned",
    config.DEFAULT_PROMPT,
    0.5,  # 修改 temperature
    config.DEFAULT_TOP_P,
    config.DEFAULT_MAX_TOKENS
)
time3 = time.time() - start
print(f"⏱️  耗时: {time3:.2f}s")
print(f"💾 来自缓存: {from_cache3}")

# 统计
print("\n" + "="*60)
print("📈 性能对比")
print("="*60)
print(f"首次处理: {time1:.2f}s")
print(f"缓存命中: {time2:.2f}s (快 {time1/time2:.1f}x ⚡)")
print(f"参数变化: {time3:.2f}s")
print("="*60)

# 显示缓存统计
cache_mgr.print_stats()