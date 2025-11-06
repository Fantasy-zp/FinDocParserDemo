"""
工具函数：PDF处理 + 模型推理 - Phase 3.1 并行版本
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
        print(f"📏 Image resized: {width}x{height} -> {new_width}x{new_height}")
    
    return image


def image_to_base64(image):
    """图片转 Base64"""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    
    image = resize_image_if_needed(image)
    
    buffer = BytesIO()
    # 使用 JPEG 压缩以减小传输大小
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffer, format='JPEG', quality=85)
    
    return base64.b64encode(buffer.getvalue()).decode()


def infer_single_image(image, model_key, prompt, temperature, top_p, max_tokens):
    """
    单张图片推理
    
    Args:
        image: PIL.Image 或文件路径
        model_key: 模型配置键
        prompt: 提示词
        temperature: 温度参数
        top_p: top_p 参数
        max_tokens: 最大 token 数
    
    Returns:
        str: Markdown 结果
    """
    # 获取模型配置
    if model_key not in config.MODELS:
        raise ValueError(f"未知的模型: {model_key}")
    
    model_config = config.MODELS[model_key]
    
    # 转换为 Base64
    img_base64 = image_to_base64(image)
    
    # 初始化客户端
    client = OpenAI(
        api_key="dummy",
        base_url=model_config["api_base"],
        timeout=120.0
    )
    
    # 调用 API
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


# ============================================
# Phase 3.1: 并行处理核心函数
# ============================================

def process_single_page_with_index(idx, image, model_key, prompt, temperature, top_p, max_tokens):
    """
    处理单页（带索引）- 供并行调用
    
    Args:
        idx: 页码索引
        image: 图像
        其他参数同 infer_single_image
    
    Returns:
        tuple: (idx, result, elapsed_time)
    """
    start_time = time.time()
    try:
        result = infer_single_image(image, model_key, prompt, temperature, top_p, max_tokens)
        elapsed = time.time() - start_time
        print(f"✅ Page {idx + 1} completed in {elapsed:.2f}s")
        return (idx, result, elapsed, None)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Page {idx + 1} failed in {elapsed:.2f}s: {str(e)}")
        return (idx, None, elapsed, str(e))


def process_images_parallel(images, model_key, prompt, temperature, top_p, max_tokens, progress=None):
    """
    并行处理多张图片（Phase 3.1 核心）
    
    Args:
        images: 图片列表
        其他参数同上
        progress: Gradio Progress 对象（可选）
    
    Returns:
        list: 按顺序的推理结果列表
    """
    total = len(images)
    
    # 单页直接处理，不启用并行
    if total < config.PARALLEL_MIN_PAGES:
        print(f"📄 Single page, using sequential processing")
        results = []
        for idx, img in enumerate(images):
            if progress is not None:
                progress((idx + 1) / total, desc=f"Processing page {idx + 1}/{total}")
            
            _, result, elapsed, error = process_single_page_with_index(
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            )
            results.append(result if error is None else f"Error: {error}")
        return results
    
    # 多页并行处理
    print(f"🚀 Parallel processing with {config.MAX_WORKERS} workers")
    
    # 初始化结果数组（保持顺序）
    results = [None] * total
    completed_count = 0
    total_time = 0
    
    # 使用线程池
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(
                process_single_page_with_index,
                idx, img, model_key, prompt, temperature, top_p, max_tokens
            ): idx
            for idx, img in enumerate(images)
        }
        
        # 收集结果
        for future in as_completed(future_to_idx):
            idx, result, elapsed, error = future.result()
            completed_count += 1
            total_time += elapsed
            
            # 保存结果
            if error is None:
                results[idx] = result
            else:
                results[idx] = f"Error on page {idx + 1}: {error}"
            
            # 更新进度
            if progress is not None:
                avg_time = total_time / completed_count
                remaining = total - completed_count
                eta = avg_time * remaining
                progress(
                    completed_count / total,
                    desc=f"Completed {completed_count}/{total} pages (ETA: {eta:.1f}s)"
                )
    
    # 统计信息
    avg_time = total_time / total
    print(f"\n{'='*60}")
    print(f"📊 Parallel Processing Statistics:")
    print(f"  Total pages: {total}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per page: {avg_time:.2f}s")
    print(f"  Workers used: {config.MAX_WORKERS}")
    print(f"{'='*60}\n")
    
    return results


def process_document(file_path, model_key, prompt, temperature, top_p, max_tokens, progress=None):
    """
    处理文档（支持并行）
    
    Args:
        file_path: 文件路径
        model_key: 模型键
        prompt: 提示词
        temperature: 温度
        top_p: top_p
        max_tokens: 最大 tokens
        progress: Gradio Progress 对象
    
    Returns:
        tuple: (图片列表, Markdown 结果)
    """
    file_path = Path(file_path)
    
    # 判断文件类型
    if file_path.suffix.lower() == '.pdf':
        if progress is not None:
            progress(0, desc="Converting PDF to images...")
        images = pdf_to_images(file_path)
        print(f"📄 PDF converted: {len(images)} pages")
    else:
        images = [Image.open(file_path)]
        print(f"🖼️  Single image loaded")
    
    # ✅ 使用并行处理（如果启用）
    if config.PARALLEL_ENABLED:
        results = process_images_parallel(
            images, model_key, prompt, temperature, top_p, max_tokens, progress
        )
    else:
        # 串行处理（保留作为备用）
        results = []
        for idx, img in enumerate(images):
            if progress is not None:
                progress((idx + 1) / len(images), desc=f"Processing page {idx + 1}/{len(images)}")
            result = infer_single_image(img, model_key, prompt, temperature, top_p, max_tokens)
            results.append(result)
    
    # 合并结果
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