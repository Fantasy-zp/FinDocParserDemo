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
                        "url": f"data:image/jpeg;base64,{img_base64}"
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
    
    return response.choices[0].message.content


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
    流式并行处理图片（Phase 3.2 核心）
    
    每完成一页就立即返回结果
    
    Args:
        images: 图片列表
        其他参数: 模型推理参数
    
    Yields:
        {
            "page_num": 1,           # 当前完成的页码
            "total_pages": 10,       # 总页数
            "result": "markdown",    # 当前页结果
            "completed": 3,          # 已完成页数
            "progress": 0.3,         # 进度 (0-1)
            "elapsed": 15.2,         # 已用时间
            "eta": 35.1,            # 预计剩余时间
            "status": "✅ Page 3/10" # 状态文本
        }
    """
    total = len(images)
    completed_count = 0
    elapsed_times = []
    start_time = time.time()
    
    # 存储结果（保持顺序）
    results = {}
    
    # 单页直接处理
    if total < config.PARALLEL_MIN_PAGES:
        for idx, img in enumerate(images):
            page_start = time.time()
            _, result, page_elapsed, error = process_single_page_with_index(
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            )
            
            completed_count += 1
            elapsed_times.append(page_elapsed)
            results[idx] = result if error is None else f"Error: {error}"
            
            # 计算 ETA
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = total - completed_count
            eta = avg_time * remaining
            
            # ✅ 立即返回当前页结果
            yield {
                "page_num": idx + 1,
                "total_pages": total,
                "result": results[idx],
                "completed": completed_count,
                "progress": completed_count / total,
                "elapsed": time.time() - start_time,
                "eta": eta,
                "status": f"✅ Page {idx + 1}/{total} completed ({page_elapsed:.1f}s)"
            }
        return
    
    # 多页并行处理
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                process_single_page_with_index,
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            ): idx
            for idx, img in enumerate(images)
        }
        
        # 实时收集结果
        for future in as_completed(future_to_idx):
            idx, result, page_elapsed, error = future.result()
            completed_count += 1
            elapsed_times.append(page_elapsed)
            
            # 保存结果
            results[idx] = result if error is None else f"Error: {error}"
            
            # 计算统计信息
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = total - completed_count
            eta = avg_time * remaining
            total_elapsed = time.time() - start_time
            
            # ✅ 立即返回当前页结果
            yield {
                "page_num": idx + 1,
                "total_pages": total,
                "result": results[idx],
                "completed": completed_count,
                "progress": completed_count / total,
                "elapsed": total_elapsed,
                "eta": eta,
                "status": f"✅ Page {idx + 1}/{total} completed ({page_elapsed:.1f}s, ETA: {eta:.1f}s)"
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
    流式处理文档（带缓存）
    
    Yields:
        (images, status, markdown, from_cache)
    """
    if not config.CACHE_ENABLED:
        # 缓存未启用，直接流式处理
        for images, status, markdown in process_document_streaming(
            file_path, model_key, prompt, temperature, top_p, max_tokens
        ):
            yield images, status, markdown, False
        return
    
    # 获取缓存管理器
    cache_mgr = get_cache_manager()
    
    # 生成缓存键
    cache_key = cache_mgr.generate_cache_key(
        file_path, model_key, prompt, temperature, top_p, max_tokens
    )
    
    # 尝试从缓存获取
    cached_result = cache_mgr.get(cache_key)
    
    if cached_result is not None:
        # ✅ 缓存命中 - 立即返回
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() == '.pdf':
            images = pdf_to_images(file_path_obj)
        else:
            images = [Image.open(file_path_obj)]
        
        markdown = cached_result["markdown"]
        pages = cached_result["metadata"]["pages"]
        
        status = f"""⚡ Loaded from cache!

📄 Pages: {pages}
🔥 Response time: <0.1s
💾 Cache hit!"""
        
        yield images, status, markdown, True
        return
    
    # ❌ 缓存未命中 - 执行流式处理
    all_results = {}
    final_images = None
    final_markdown = ""
    
    for images, status, markdown in process_document_streaming(
        file_path, model_key, prompt, temperature, top_p, max_tokens
    ):
        final_images = images
        final_markdown = markdown
        yield images, status, markdown, False
    
    # 处理完成，保存到缓存
    if final_images is not None:
        result = {
            "markdown": final_markdown,
            "metadata": {
                "pages": len(final_images),
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