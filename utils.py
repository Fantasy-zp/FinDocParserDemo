"""
工具函数：PDF处理 + 模型推理 - Phase 3.2 流式版本
"""
import base64
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI
import config
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Generator, Dict, Any, List, Tuple
from cache_manager import get_cache_manager
import requests
import json
import threading


def pdf_to_images(pdf_path):
    """PDF 转图片"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(config.PDF_DPI / 72, config.PDF_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    doc.close()
    return images


def resize_image_if_needed(image):
    """调整图像大小"""
    width, height = image.size
    pixels = width * height
    
    if pixels > config.IMAGE_MAX_PIXELS:
        scale = (config.IMAGE_MAX_PIXELS / pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


def image_to_base64(image):
    """图片转 Base64"""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    
    image = resize_image_if_needed(image)
    
    buffer = BytesIO()
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffer, format='JPEG', quality=85)
    
    return base64.b64encode(buffer.getvalue()).decode()


def infer_single_image(image, model_key, prompt, temperature, top_p, max_tokens):
    """单张图片推理"""
    if model_key not in config.MODELS:
        raise ValueError(f"未知的模型: {model_key}")
    
    model_config = config.MODELS[model_key]
    img_base64 = image_to_base64(image)
    
    client = OpenAI(
        api_key="dummy",
        base_url=model_config["api_base"],
        timeout=120.0
    )
    
    response = client.chat.completions.create(
        model=model_config["model_id"],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={
            'repetition_penalty': 1.0,
            'top_k': 50,
            'skip_special_tokens': True,
        }
    )
    # 获取返回内容
    result = response.choices[0].message.content

    # ✅ 添加 JSON 格式判断和提取
    result = extract_markdown_from_result(result)
    
    return result

# ✅ 添加新函数：提取 markdown 内容
def extract_markdown_from_result(result: str) -> str:
    """
    从 API 返回结果中提取 markdown 内容（完整增强版）
    
    支持：
    1. 完整 JSON → json.loads 解析
    2. 不完整 JSON → 正则提取
    3. 纯 Markdown → 直接返回
    """
    import json
    import re
    
    original_result = result
    result = result.strip()
    
    # 快速排除：明显不是 JSON
    if not result.startswith('{'):
        return original_result
    
    # 检查是否包含 natural_text 字段
    if '"natural_text"' not in result:
        print("ℹ️  JSON 格式但不包含 natural_text 字段")
        return original_result
    
    # ============================================
    # 尝试 1：完整 JSON 解析
    # ============================================
    try:
        data = json.loads(result)
        
        if isinstance(data, dict) and 'natural_text' in data:
            markdown_text = data['natural_text']
            print(f"✅ 从完整 JSON 提取 natural_text (长度: {len(markdown_text)} 字符)")
            return f"{markdown_text}\n\n<!-- RAW_OUTPUT_START\n{original_result}\nRAW_OUTPUT_END -->"
        else:
            print(f"⚠️  JSON 解析成功但结构不符合预期")
            return original_result
            
    except json.JSONDecodeError as e:
        # ============================================
        # 尝试 2：从不完整 JSON 中正则提取
        # ============================================
        print(f"⚠️  JSON 不完整，尝试正则提取...")
        print(f"   错误信息: {str(e)}")
        print(f"   最后 50 字符: ...{result[-50:]}")
        
        # 用正则提取 natural_text 的值
        extracted = extract_natural_text_by_regex(result)
        
        if extracted:
            print(f"⚡ 成功从不完整 JSON 中提取 {len(extracted)} 字符")
            
            # 添加截断警告
            # warning = "\n\n---\n⚠️ **警告**：输出被截断（达到 Max Tokens 限制），内容可能不完整。建议增加 Max Tokens 参数。"
            
            return f"{extracted}\n\n<!-- RAW_OUTPUT_START\n{original_result}\nRAW_OUTPUT_END -->"
        else:
            print(f"❌ 无法从不完整 JSON 中提取 natural_text")
            return f"⚠️ **解析失败**：输出被截断且无法提取内容。\n\n**建议**：\n1. 增加 Max Tokens 到 16384\n2. 简化文档或分页处理\n\n**原始输出**：\n```\n{original_result[:500]}...\n```"


def extract_natural_text_by_regex(incomplete_json: str) -> str:
    """
    用正则从不完整的 JSON 中提取 natural_text 的值
    
    Args:
        incomplete_json: 不完整的 JSON 字符串
    
    Returns:
        提取的 markdown 内容（可能不完整）
    """
    import re
    
    # 匹配模式（按优先级尝试）
    patterns = [
        # 模式 1：完整的值（带结束引号和逗号/右括号）
        r'"natural_text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        
        # 模式 2：不完整的值（没有结束引号，直到字符串末尾）
        r'"natural_text"\s*:\s*"((?:[^"\\]|\\.)*?)(?:$|")',
    ]
    
    for i, pattern in enumerate(patterns, 1):
        match = re.search(pattern, incomplete_json, re.DOTALL)
        
        if match:
            content = match.group(1)
            
            # 处理 JSON 转义字符
            content = unescape_json_string(content)
            
            print(f"   ✅ 正则模式 {i} 匹配成功")
            return content.strip()
    
    return ""


def unescape_json_string(s: str) -> str:
    """处理 JSON 字符串中的转义字符"""
    # 按顺序处理（顺序很重要）
    replacements = [
        ('\\n', '\n'),   # 换行
        ('\\t', '\t'),   # 制表符
        ('\\r', '\r'),   # 回车
        ('\\"', '"'),    # 引号
        ('\\/', '/'),    # 斜杠
        ('\\\\', '\\'),  # 反斜杠（最后处理）
    ]
    
    for old, new in replacements:
        s = s.replace(old, new)
    
    return s

# def extract_markdown_from_result(result: str) -> str:
#     """增强版：支持多种提取策略"""
#     import json
    
#     result = result.strip()
    
#     # 策略 1：JSON 格式
#     if result.startswith('{') and result.endswith('}'):
#         try:
#             data = json.loads(result)
            
#             if isinstance(data, dict):
#                 # 优先级：natural_text > text > content > markdown
#                 for key in ['natural_text', 'text', 'content', 'markdown']:
#                     if key in data:
#                         print(f"✅ 从 JSON 提取字段: {key}")
#                         return data[key]
#         except json.JSONDecodeError:
#             pass
    
#     # 策略 2：Markdown 代码块
#     if '```markdown' in result:
#         import re
#         match = re.search(r'```markdown\n(.*?)\n```', result, re.DOTALL)
#         if match:
#             print("✅ 从 markdown 代码块提取")
#             return match.group(1)
    
#     # 默认：返回原始结果
#     return result

def process_single_page_with_index(idx, image, model_key, prompt, temperature, top_p, max_tokens):
    """处理单页（带索引）"""
    start_time = time.time()
    try:
        result = infer_single_image(image, model_key, prompt, temperature, top_p, max_tokens)
        elapsed = time.time() - start_time
        return (idx, result, elapsed, None)
    except Exception as e:
        elapsed = time.time() - start_time
        return (idx, None, elapsed, str(e))


# ============================================
# Phase 3.2: 流式处理核心函数
# ============================================

def process_images_streaming(
    images: List[Image.Image],
    model_key: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Generator[Dict[str, Any], None, None]:
    """
    流式并行处理图片（Phase 3.4 优化版 - 单页实时反馈增强）
    
    单页处理时会实时更新已用时间
    """
    import threading
    
    total = len(images)
    completed_count = 0
    elapsed_times = []
    start_time = time.time()
    
    # 存储结果（保持顺序）
    results = {}
    
    # 单页直接处理（✅ 添加实时进度更新）
    if total < config.PARALLEL_MIN_PAGES:
        for idx, img in enumerate(images):
            # 用于线程间通信
            result_container = {"result": None, "elapsed": 0, "error": None, "done": False}
            
            # ✅ 在独立线程中执行推理
            def inference_thread():
                page_start = time.time()
                try:
                    _, result, page_elapsed, error = process_single_page_with_index(
                        idx, img, model_key, prompt, temperature, top_p, max_tokens
                    )
                    result_container["result"] = result
                    result_container["elapsed"] = page_elapsed
                    result_container["error"] = error
                except Exception as e:
                    result_container["error"] = str(e)
                finally:
                    result_container["done"] = True
            
            # 启动推理线程
            thread = threading.Thread(target=inference_thread, daemon=True)
            thread.start()
            
            # ✅ 主线程定期更新状态（每0.5秒）
            page_start_time = time.time()
            while not result_container["done"]:
                elapsed = time.time() - page_start_time
                
                # 模拟进度（基于时间估算）
                # 假设平均每页 5-10 秒，用脉搏动画
                pulse = int((elapsed * 2) % 20)  # 0-19 循环
                progress_bar = "█" * pulse + "░" * (20 - pulse)
                
                yield {
                    "page_num": idx + 1,
                    "total_pages": total,
                    "result": None,
                    "completed": completed_count,
                    "progress": 0,
                    "elapsed": elapsed,
                    "eta": 0,
                    "status": f"⏳ 正在处理第 {idx + 1}/{total} 页..."
                }
                
                time.sleep(0.5)  # 每0.5秒更新一次
            
            # ✅ 等待线程完成
            thread.join(timeout=1)
            
            # 处理结果
            completed_count += 1
            elapsed_times.append(result_container["elapsed"])
            
            if result_container["error"] is None:
                results[idx] = result_container["result"]
            else:
                results[idx] = f"Error: {result_container['error']}"
            
            # ✅ 返回完成状态
            yield {
                "page_num": idx + 1,
                "total_pages": total,
                "result": results[idx],
                "completed": completed_count,
                "progress": 1.0,
                "elapsed": result_container["elapsed"],
                "eta": 0,
                "status": f"✅ 第 {idx + 1}/{total} 页完成 ({result_container['elapsed']:.1f}s)"
            }
        return
    
    # 多页并行处理（保持不变）
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                process_single_page_with_index,
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            ): idx
            for idx, img in enumerate(images)
        }
        
        for future in as_completed(future_to_idx):
            idx, result, page_elapsed, error = future.result()
            completed_count += 1
            elapsed_times.append(page_elapsed)
            
            results[idx] = result if error is None else f"Error on page {idx + 1}: {error}"
            
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = total - completed_count
            eta = avg_time * remaining
            total_elapsed = time.time() - start_time
            
            yield {
                "page_num": idx + 1,
                "total_pages": total,
                "result": results[idx],
                "completed": completed_count,
                "progress": completed_count / total,
                "elapsed": total_elapsed,
                "eta": eta,
                "status": f"✅ 第 {idx + 1}/{total} 页完成 ({page_elapsed:.1f}s, 预计剩余: {eta:.1f}s)"
            }

def merge_results_ordered(results: Dict[int, str], total: int) -> str:
    """
    按顺序合并结果
    
    Args:
        results: {页码索引: markdown结果}
        total: 总页数
    
    Returns:
        合并后的 markdown
    """
    ordered_results = []
    for i in range(total):
        if i in results:
            ordered_results.append(f"## Page {i + 1}\n\n{results[i]}")
        else:
            ordered_results.append(f"## Page {i + 1}\n\n⏳ Processing...")
    
    return "\n\n---\n\n".join(ordered_results)


def process_document_streaming(
    file_path: str,
    model_key: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Generator[Tuple[List[Image.Image], str, str], None, None]:
    """
    流式处理文档（Phase 3.2 主接口）
    
    每完成一页就返回当前状态
    
    Args:
        file_path: 文件路径
        其他参数: 推理参数
    
    Yields:
        (images, status, markdown)
        - images: 已处理的图片列表
        - status: 状态文本
        - markdown: 当前累积的结果
    """
    file_path = Path(file_path)
    
    # 加载图片
    if file_path.suffix.lower() == '.pdf':
        images = pdf_to_images(file_path)
    else:
        images = [Image.open(file_path)]
    
    total = len(images)
    
    # 初始状态
    yield (
        images,
        f"📄 Loaded {total} page(s), starting processing...",
        ""
    )
    
    # 收集所有结果
    all_results = {}
    
    # 流式处理
    for update in process_images_streaming(
        images, model_key, prompt, temperature, top_p, max_tokens
    ):
        page_idx = update["page_num"] - 1
        all_results[page_idx] = update["result"]
        
        # 构建状态文本
        progress_bar = "█" * int(update["progress"] * 20) + "░" * (20 - int(update["progress"] * 20))
        status = f"""🔄 Processing: {update['completed']}/{update['total_pages']} pages

{progress_bar} {update['progress']*100:.1f}%

⏱️  Elapsed: {update['elapsed']:.1f}s
⏰ ETA: {update['eta']:.1f}s

{update['status']}"""
        
        # 合并当前结果
        if total == 1:
            markdown = all_results.get(0, "")
        else:
            markdown = merge_results_ordered(all_results, total)
        
        # ✅ 返回当前状态
        yield (images, status, markdown)
    
    # 最终状态
    final_markdown = merge_results_ordered(all_results, total) if total > 1 else all_results.get(0, "")
    
    yield (
        images,
        f"✅ Completed! {total} page(s) processed successfully.",
        final_markdown
    )


# ============================================
# 保留原有接口（向下兼容）
# ============================================

def process_images_parallel(images, model_key, prompt, temperature, top_p, max_tokens, progress=None):
    """并行处理（非流式，保留兼容）"""
    total = len(images)
    
    if total < config.PARALLEL_MIN_PAGES:
        results = []
        for idx, img in enumerate(images):
            if progress is not None:
                progress((idx + 1) / total, desc=f"Processing page {idx + 1}/{total}")
            
            _, result, elapsed, error = process_single_page_with_index(
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            )
            results.append(result if error is None else f"Error: {error}")
        return results
    
    results = [None] * total
    completed_count = 0
    total_time = 0
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                process_single_page_with_index,
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            ): idx
            for idx, img in enumerate(images)
        }
        
        for future in as_completed(future_to_idx):
            idx, result, elapsed, error = future.result()
            completed_count += 1
            total_time += elapsed
            
            if error is None:
                results[idx] = result
            else:
                results[idx] = f"Error on page {idx + 1}: {error}"
            
            if progress is not None:
                avg_time = total_time / completed_count
                remaining = total - completed_count
                eta = avg_time * remaining
                progress(
                    completed_count / total,
                    desc=f"Completed {completed_count}/{total} pages (ETA: {eta:.1f}s)"
                )
    
    return results


def process_document(file_path, model_key, prompt, temperature, top_p, max_tokens, progress=None):
    """处理文档（非流式，保留兼容）"""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        if progress is not None:
            progress(0, desc="Converting PDF to images...")
        images = pdf_to_images(file_path)
    else:
        images = [Image.open(file_path)]
    
    if config.PARALLEL_ENABLED:
        results = process_images_parallel(
            images, model_key, prompt, temperature, top_p, max_tokens, progress
        )
    else:
        results = []
        for idx, img in enumerate(images):
            if progress is not None:
                progress((idx + 1) / len(images), desc=f"Processing page {idx + 1}/{len(images)}")
            result = infer_single_image(img, model_key, prompt, temperature, top_p, max_tokens)
            results.append(result)
    
    if len(results) > 1:
        markdown = "\n\n---\n\n".join([
            f"## Page {i + 1}\n\n{result}" 
            for i, result in enumerate(results)
        ])
    else:
        markdown = results[0]
    
    return images, markdown


def validate_file(file_path):
    """验证文件"""
    if not file_path:
        return False, "请上传文件"
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False, "文件不存在"
    
    if file_path.suffix.lower() not in config.ALLOWED_FILE_TYPES:
        return False, f"不支持的文件类型，仅支持：{', '.join(config.ALLOWED_FILE_TYPES)}"
    
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"文件过大（{size_mb:.1f}MB），最大支持 {config.MAX_FILE_SIZE_MB}MB"
    
    return True, ""


def get_model_choices():
    """获取模型选择列表"""
    return [model["name"] for model in config.MODELS.values()]


def get_model_key_from_name(model_name):
    """从显示名称获取模型键"""
    for key, model in config.MODELS.items():
        if model["name"] == model_name:
            return key
    return config.DEFAULT_MODEL


def test_model_connection(model_key):
    """测试模型 API 连接"""
    try:
        model_config = config.MODELS[model_key]
        client = OpenAI(
            api_key="dummy",
            base_url=model_config["api_base"],
            timeout=5.0
        )
        
        client.models.list()
        return True, f"✅ {model_config['name']} 连接正常"
    
    except Exception as e:
        return False, f"❌ {model_config['name']} 连接失败: {str(e)}"

# ============================================
# Phase 3.3: 缓存配置
# ============================================

def process_document_with_cache(
    file_path: str,
    model_key: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Tuple[List[Image.Image], str, bool]:
    """
    处理文档（带缓存）
    
    Returns:
        (images, markdown, from_cache)
        - images: 图片列表
        - markdown: 结果
        - from_cache: 是否来自缓存
    """
    if not config.CACHE_ENABLED:
        # 缓存未启用，直接处理
        images, markdown = process_document(
            file_path, model_key, prompt, temperature, top_p, max_tokens
        )
        return images, markdown, False
    
    # 获取缓存管理器
    cache_mgr = get_cache_manager()
    
    # 生成缓存键
    cache_key = cache_mgr.generate_cache_key(
        file_path, model_key, prompt, temperature, top_p, max_tokens
    )
    
    # 尝试从缓存获取
    cached_result = cache_mgr.get(cache_key)
    
    if cached_result is not None:
        # 缓存命中
        # 重新加载图片
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.pdf':
            images = pdf_to_images(file_path)
        else:
            images = [Image.open(file_path)]
        
        markdown = cached_result["markdown"]
        return images, markdown, True
    
    # 缓存未命中，执行推理
    images, markdown = process_document(
        file_path, model_key, prompt, temperature, top_p, max_tokens
    )
    
    # 保存到缓存
    result = {
        "markdown": markdown,
        "metadata": {
            "pages": len(images),
            "model": model_key,
            "timestamp": time.time()
        }
    }
    
    cache_mgr.set(
        cache_key,
        result,
        Path(file_path).name,
        model_key,
        temperature,
        top_p,
        max_tokens
    )
    
    return images, markdown, False

def is_valid_result(markdown: str) -> bool:
    """
    检查解析结果是否有效（不包含错误）
    
    规则：
    1. 内容不能为空
    2. 内容长度 >= 10 字符
    3. 不包含错误关键词
    4. 错误数量不能过多
    
    Args:
        markdown: 解析得到的 markdown 内容
    
    Returns:
        True: 有效结果，可以缓存
        False: 包含错误，不应缓存
    """
    # 检查 1：非空
    if not markdown:
        print("   ⚠️  内容为空")
        return False
    
    # 检查 2：最小长度
    content = markdown.strip()
    if len(content) < 10:
        print(f"   ⚠️  内容过短 (只有 {len(content)} 字符)")
        return False
    
    # 检查 3：错误关键词（不区分大小写）
    error_patterns = [
        "error:",
        "exception:",
        "failed:",
        "connection error",
        "timeout",
        "api error",
        "invalid response",
        "<!-- error:",  # HTML 注释中的错误
    ]
    
    content_lower = content.lower()
    
    for pattern in error_patterns:
        if pattern in content_lower:
            print(f"   ⚠️  检测到错误标识: '{pattern}'")
            return False
    
    # 检查 4：错误密度（防止多页都失败）
    error_count = content_lower.count("error")
    total_length = len(content)
    
    if error_count > 0:
        error_density = error_count / (total_length / 1000)  # 每1000字符的错误数
        if error_density > 0.5:  # 如果每1000字符有超过0.5个error
            print(f"   ⚠️  错误密度过高 (error 出现 {error_count} 次)")
            return False
    
    # 通过所有检查
    return True
# def is_valid_result(markdown: str) -> bool:
#     """增强版：更严格的验证"""
#     if not markdown or len(markdown.strip()) < 10:
#         return False
    
#     content = markdown.strip()
#     content_lower = content.lower()
    
#     # 1. 严格的错误检查
#     strict_errors = [
#         "connection error",
#         "timeout",
#         "api error",
#         "authentication failed",
#         "rate limit exceeded",
#     ]
    
#     for error in strict_errors:
#         if error in content_lower:
#             return False
    
#     # 2. 检查是否有实质内容（不只是错误信息）
#     # 至少应该包含一些正常的 markdown 元素
#     markdown_indicators = ["#", "##", "table", "```", "-", "*"]
#     has_markdown = any(indicator in content for indicator in markdown_indicators)
    
#     if not has_markdown and "error" in content_lower:
#         print("   ⚠️  只包含错误信息，没有有效内容")
#         return False
    
#     # 3. 检查 JSON 格式的错误（如果使用了 JSON 格式）
#     if content.startswith('{') and '"error"' in content_lower:
#         return False
    
#     return True

def extract_error_reason(markdown: str) -> str:
    """
    从错误内容中提取错误原因
    
    Args:
        markdown: 包含错误的 markdown 内容
    
    Returns:
        简短的错误描述
    """
    if not markdown:
        return "未知错误"
    
    content_lower = markdown.lower()
    
    # 按优先级检查错误类型
    if "connection" in content_lower or "connect" in content_lower:
        return "网络连接失败"
    elif "timeout" in content_lower or "timed out" in content_lower:
        return "请求超时"
    elif "authentication" in content_lower or "unauthorized" in content_lower:
        return "认证失败"
    elif "rate limit" in content_lower or "too many requests" in content_lower:
        return "请求频率超限"
    elif "api error" in content_lower or "api_error" in content_lower:
        return "API 服务错误"
    elif "invalid" in content_lower:
        return "无效的请求"
    elif "not found" in content_lower or "404" in content_lower:
        return "API 地址错误"
    elif "server error" in content_lower or "500" in content_lower:
        return "服务器错误"
    else:
        # 尝试提取 Error: 后面的内容
        if "error:" in content_lower:
            try:
                error_start = content_lower.index("error:") + 6
                error_msg = markdown[error_start:error_start + 50].strip()
                # 取第一行或前30个字符
                error_msg = error_msg.split('\n')[0][:30]
                return error_msg if error_msg else "解析错误"
            except:
                pass
        
        return "解析错误"

def process_document_streaming_with_cache(
    file_path: str,
    model_key: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Generator[Tuple[List[Image.Image], str, str, bool], None, None]:
    """
    流式处理文档（支持多种模型类型 - Phase 3.5）
    
    Yields:
        (images, status, markdown, from_cache)
    """
    model_info = config.MODELS.get(model_key)
    if not model_info:
        yield (None, f"❌ 未知模型: {model_key}", "", False)
        return
    
    model_type = model_info.get("type", "openai")
    
    # 根据模型类型选择处理方式
    if model_type == "custom":
        # ✅ 自定义 API（跨页合并模型）
        yield from process_with_custom_model(file_path, model_key)
        return

    # ============================================
    # 原有代码（OpenAI 兼容模型）
    # ============================================
    if not config.CACHE_ENABLED:
        for images, status, markdown in process_document_streaming(
            file_path, model_key, prompt, temperature, top_p, max_tokens
        ):
            yield images, status, markdown, False
        return
    
    cache_mgr = get_cache_manager()
    cache_key = cache_mgr.generate_cache_key(
        file_path, model_key, prompt, temperature, top_p, max_tokens
    )
    
    # 尝试从缓存获取
    cached_result = cache_mgr.get(cache_key)
    
    if cached_result is not None:
        # 缓存命中
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() == '.pdf':
            images = pdf_to_images(file_path_obj)
        else:
            images = [Image.open(file_path_obj)]
        
        markdown = cached_result["markdown"]
        pages = cached_result["metadata"]["pages"]
        
        status = f"""⚡ 从缓存加载！

████████████████████ 100%

📄 页数: {pages} 页
🔥 响应时间: <0.1s
💾 缓存命中！"""
        
        yield images, status, markdown, True
        return
    
    # 缓存未命中 - 执行流式处理
    file_path_obj = Path(file_path)
    
    if file_path_obj.suffix.lower() == '.pdf':
        images = pdf_to_images(file_path_obj)
    else:
        images = [Image.open(file_path_obj)]
    
    total = len(images)
    start_time = time.time()
    
    # 初始状态
    initial_status = f"📄 已加载 {total} 页，开始处理..."
    yield (images, initial_status, "", False)
    
    # 收集所有结果
    all_results = {}
    
    # 流式处理
    for update in process_images_streaming(
        images, model_key, prompt, temperature, top_p, max_tokens
    ):
        page_idx = update["page_num"] - 1
        
        if update["result"] is not None:
            all_results[page_idx] = update["result"]
        
        # 构建状态文本
        if total == 1:
            # 单页的状态显示
            if update["completed"] == 0:
                # 处理中（实时更新时间）
                pulse = int((update["elapsed"] * 2) % 20)
                progress_bar = "█" * pulse + "░" * (20 - pulse)
                
                status = f"""⏳ 正在处理...

{progress_bar}

⏱️  已用时间: {update['elapsed']:.1f}s

{update['status']}"""
            else:
                # 完成
                progress_bar = "█" * 20
                status = f"""✅ 处理完成！

{progress_bar} 100%

⏱️  处理时间: {update['elapsed']:.1f}s

{update['status']}"""
        else:
            # 多页的状态显示
            progress_bar = "█" * int(update["progress"] * 20) + "░" * (20 - int(update["progress"] * 20))
            status = f"""🔄 处理中: {update['completed']}/{update['total_pages']} 页

{progress_bar} {update['progress']*100:.1f}%

⏱️  已用时间: {update['elapsed']:.1f}s
⏰ 预计剩余: {update['eta']:.1f}s

{update['status']}"""
        
        # 合并当前结果
        if total == 1:
            if 0 in all_results:
                markdown = all_results[0]
            else:
                markdown = "⏳ 正在处理中..."
        else:
            markdown = merge_results_ordered(all_results, total)
        
        yield (images, status, markdown, False)
    
    # ============================================
    # 处理完成 - 构建最终状态（详细错误版）
    # ============================================
    final_markdown = merge_results_ordered(all_results, total) if total > 1 else all_results.get(0, "")
    total_elapsed = time.time() - start_time
    
    # ✅ 验证结果是否有效
    is_valid = is_valid_result(final_markdown) if final_markdown else False
    
    # ✅ 根据验证结果构建不同的状态信息
    if is_valid:
        # ========== 成功情况 ==========
        if total == 1:
            final_status = f"""✅ 解析完成！

████████████████████ 100%

📄 页数: 1 页
⏱️  处理时间: {total_elapsed:.1f}s
💾 已保存到缓存"""
        else:
            final_status = f"""✅ 解析完成！

████████████████████ 100%

📄 总页数: {total} 页
⏱️  总耗时: {total_elapsed:.1f}s
💾 已保存到缓存"""
        
        # 保存到缓存
        if images is not None:
            result = {
                "markdown": final_markdown,
                "metadata": {
                    "pages": len(images),
                    "model": model_key,
                    "timestamp": time.time()
                }
            }
            
            cache_mgr.set(
                cache_key,
                result,
                file_path_obj.name,
                model_key,
                temperature,
                top_p,
                max_tokens
            )
            print(f"✅ 有效结果已保存到缓存")
    else:
        # ========== 失败情况（详细错误提示）==========
        error_reason = extract_error_reason(final_markdown)
        
        if total == 1:
            final_status = f"""❌ 解析失败！

████████████████████ 100%

📄 页数: 1 页
⏱️  处理时间: {total_elapsed:.1f}s
⚠️ 错误原因: {error_reason}
💡 建议: 检查网络连接或更换模型"""
        else:
            final_status = f"""❌ 解析失败！

████████████████████ 100%

📄 总页数: {total} 页
⏱️  总耗时: {total_elapsed:.1f}s
⚠️ 错误原因: {error_reason}
💡 建议: 检查网络连接或更换模型"""
        
        print(f"⚠️  解析失败: {error_reason}，跳过缓存")
    
    # 返回最终结果
    yield (images, final_status, final_markdown, False)


def get_cache_stats():
    """获取缓存统计（供界面调用）"""
    if not config.CACHE_ENABLED:
        return "Cache disabled"
    
    cache_mgr = get_cache_manager()
    stats = cache_mgr.get_stats()
    
    return f"""📊 Cache Statistics:
    
💾 Memory: {stats['memory_cache_size']} entries
💽 Disk: {stats['disk_cache_count']} entries ({stats['disk_cache_size_mb']:.1f}MB)

📈 Performance:
  Total requests: {stats['total_requests']}
  Memory hits: {stats['memory_hits']} ⚡
  Disk hits: {stats['disk_hits']} 💾
  Misses: {stats['misses']} ❌
  
🎯 Hit rate: {stats['hit_rate']}"""


def clear_cache():
    """清空缓存（供界面调用）"""
    if not config.CACHE_ENABLED:
        return "Cache disabled"
    
    cache_mgr = get_cache_manager()
    cache_mgr.clear_all()
    return "✅ Cache cleared successfully!"

# ============================================
# Phase 3.5: 自定义模型支持（跨页合并）
# ============================================

import json

def infer_with_custom_api(
    pdf_path: str,  # 虽然叫 pdf_path，但也支持图片
    api_base: str,
    timeout: int = 300
) -> str:
    """
    调用自定义 API 进行文档解析（支持 PDF 和图片）
    
    Args:
        pdf_path: 文件路径（PDF 或图片）
        api_base: API 地址（如 http://127.0.0.1:8002）
        timeout: 超时时间（秒）
    
    Returns:
        解析结果（markdown + 隐藏的原始 JSON）
    """
    try:
        parse_url = f"{api_base}/parse"
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        # ✅ 修改 3：根据文件类型设置 MIME type
        suffix = pdf_file.suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        mime_type = mime_types.get(suffix, 'application/octet-stream')
        
        print(f"📤 上传文件到跨页合并 API: {parse_url}")
        print(f"   文件: {pdf_file.name} ({pdf_file.stat().st_size / 1024:.1f}KB)")
        print(f"   类型: {mime_type}")  # ✅ 显示文件类型
        
        # 发送请求
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, mime_type)}  # ✅ 使用正确的 MIME type
            response = requests.post(parse_url, files=files, timeout=timeout)
        
        # 检查响应
        response.raise_for_status()
        
        # 解析 JSON 响应
        result = response.json()
        
        # 保存原始 JSON
        original_json = json.dumps(result, ensure_ascii=False, indent=2)
        
        # 根据实际返回格式提取内容
        if result.get('success') and 'result' in result:
            document_text = result['result'].get('document_text')
            num_pages = result['result'].get('num_pages', 0)
            
            if document_text:
                print(f"✅ 跨页模型解析完成")
                print(f"   页数: {num_pages}")
                print(f"   内容长度: {len(document_text)} 字符")
                print(f"   原始 JSON 长度: {len(original_json)} 字符")
                
                combined = f"{document_text}\n\n<!-- RAW_OUTPUT_START\n{original_json}\nRAW_OUTPUT_END -->"
                return combined
            else:
                error_msg = "API 返回的 document_text 为空"
                print(f"⚠️  {error_msg}")
                return f"<!-- Error: {error_msg} -->\n\n<!-- RAW_OUTPUT_START\n{original_json}\nRAW_OUTPUT_END -->"
        else:
            error_msg = result.get('error', '未知错误')
            print(f"❌ API 返回失败: {error_msg}")
            return f"<!-- Error: {error_msg} -->\n\n<!-- RAW_OUTPUT_START\n{original_json}\nRAW_OUTPUT_END -->"
        
    except requests.exceptions.Timeout:
        error_msg = f"请求超时（超过 {timeout} 秒）"
        print(f"⚠️  {error_msg}")
        return f"<!-- Error: {error_msg} -->"
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"连接失败 - {str(e)}"
        print(f"⚠️  {error_msg}")
        return f"<!-- Error: {error_msg} -->"
        
    except Exception as e:
        error_msg = f"解析失败: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return f"<!-- Error: {error_msg} -->"


def check_custom_api_health(api_base: str) -> bool:
    """
    检查自定义 API 健康状态
    
    Args:
        api_base: API 地址
    
    Returns:
        True: 健康，False: 不可用
    """
    try:
        health_url = f"{api_base}/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                print(f"✅ 跨页模型 API 健康: {api_base}")
                return True
            else:
                print(f"⚠️  API 状态异常: {data}")
                return False
        else:
            print(f"⚠️  API 返回状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API 健康检查失败: {e}")
        return False


def process_with_custom_model(
    file_path: str,
    model_key: str
) -> Generator[Tuple[List[Image.Image], str, str, bool], None, None]:
    """使用自定义 API 处理文档（支持 PDF 和图片）"""
    file_path_obj = Path(file_path)
    model_info = config.MODELS[model_key]
    
    # ============================================
    # 1. 检查缓存
    # ============================================
    if config.CACHE_ENABLED:
        cache_mgr = get_cache_manager()
        cache_key = cache_mgr.generate_cache_key(
            file_path, model_key, "", 0, 0, 0
        )
        
        cached_result = cache_mgr.get(cache_key)
        if cached_result is not None:
            # 缓存命中
            try:
                # ✅ 修改 1：根据文件类型加载预览
                if file_path_obj.suffix.lower() == '.pdf':
                    images = pdf_to_images(file_path_obj)
                else:
                    images = [Image.open(file_path_obj)]
            except Exception as e:
                yield (None, f"❌ 文件加载失败: {str(e)}", "", False)
                return
            
            markdown = cached_result["markdown"]
            pages = cached_result["metadata"]["pages"]
            
            status = f"""⚡ 从缓存加载！

████████████████████ 100%

📄 页数: {pages} 页
🔥 响应时间: <0.1s
💾 缓存命中！"""
            
            yield images, status, markdown, True
            return
    
    # ============================================
    # 2. 转换文件为预览图片
    # ============================================
    try:
        # ✅ 修改 2：根据文件类型加载预览
        if file_path_obj.suffix.lower() == '.pdf':
            images = pdf_to_images(file_path_obj)
        else:
            images = [Image.open(file_path_obj)]
    except Exception as e:
        yield (None, f"❌ 文件加载失败: {str(e)}", "", False)
        return
    
    total = len(images)
    start_time = time.time()
    
    # 初始状态
    initial_status = f"""📄 已加载 {total} 页文档

⏳ 正在解析整个文档（支持跨页内容自动合并）...

💡 提示：此模型会自动处理跨页内容，无需逐页解析"""
    
    yield (images, initial_status, "", False)
    
    # ============================================
    # 3. 带心跳的 API 调用（保持不变）
    # ============================================
    result_container = {
        "done": False,
        "result": None,
        "error": None
    }
    
    def api_call_thread():
        try:
            result_container["result"] = infer_with_custom_api(
                file_path, 
                model_info["api_base"]
            )
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True
    
    thread = threading.Thread(target=api_call_thread, daemon=True)
    thread.start()
    
    heartbeat_interval = 1.0
    
    while not result_container["done"]:
        current_time = time.time()
        elapsed = current_time - start_time
        
        pulse_position = int((elapsed / heartbeat_interval) % 20)
        progress_bar = "█" * pulse_position + "░" * (20 - pulse_position)
        
        if elapsed < 10:
            hint = "🔍 正在分析文档结构..."
        elif elapsed < 20:
            hint = "📊 正在识别文本和表格..."
        elif elapsed < 30:
            hint = "🔗 正在合并跨页内容..."
        elif elapsed < 45:
            hint = "✨ 正在优化输出格式..."
        else:
            hint = "⏳ 复杂文档需要更多时间，请稍候..."
        
        status = f"""⏳ 正在解析整个文档...

{progress_bar}

⏱️  已用时间: {elapsed:.1f}s
📄 总页数: {total} 页
{hint}"""
        
        yield (images, status, "", False)
        time.sleep(heartbeat_interval)
    
    # ============================================
    # 4. 处理结果（保持不变）
    # ============================================
    elapsed = time.time() - start_time
    
    if result_container["error"]:
        error_status = f"""❌ 解析失败！

████████████████████ 100%

📄 总页数: {total} 页
⏱️  处理时间: {elapsed:.1f}s
⚠️  错误原因: {result_container['error']}

💡 建议: 
  1. 检查 API 服务状态
  2. 确认文档格式正确
  3. 查看日志获取详细信息"""
        
        yield (images, error_status, "", False)
        return
    
    markdown = result_container["result"]
    is_valid = is_valid_result(markdown) if markdown else False
    
    if is_valid:
        final_status = f"""✅ 解析完成！

████████████████████ 100%

📄 总页数: {total} 页
⏱️  处理时间: {elapsed:.1f}s
💾 已保存到缓存"""
        
        if config.CACHE_ENABLED:
            result = {
                "markdown": markdown,
                "metadata": {
                    "pages": total,
                    "model": model_key,
                    "timestamp": time.time()
                }
            }
            cache_mgr.set(
                cache_key, result, file_path_obj.name,
                model_key, 0, 0, 0
            )
            print(f"✅ 有效结果已保存到缓存")
        
    else:
        error_reason = extract_error_reason(markdown)
        final_status = f"""❌ 解析失败！

████████████████████ 100%

📄 总页数: {total} 页
⏱️  处理时间: {elapsed:.1f}s
⚠️  错误原因: {error_reason}
💡 建议: 
  1. 检查 API 服务状态
  2. 确认文档格式正确
  3. 查看日志获取详细信息"""
        
        print(f"⚠️  解析失败: {error_reason}")
    
    yield (images, final_status, markdown, False)