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


def process_document_streaming_with_cache(
    file_path: str,
    model_key: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Generator[Tuple[List[Image.Image], str, str, bool], None, None]:
    """
    流式处理文档（带缓存 - Phase 3.4 优化版 - 实时进度更新）
    
    Yields:
        (images, status, markdown, from_cache)
    """
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
        
        # ✅ 构建状态文本（实时更新版）
        if total == 1:
            # 单页的状态显示
            if update["completed"] == 0:
                # ✅ 处理中（实时更新时间）
                # 使用脉搏动画而不是固定进度
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
    
    # 处理完成
    final_markdown = merge_results_ordered(all_results, total) if total > 1 else all_results.get(0, "")
    total_elapsed = time.time() - start_time
    
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