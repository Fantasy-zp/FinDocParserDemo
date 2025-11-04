"""
工具函数：PDF处理 + 模型推理
"""
import base64
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI
import config
from io import BytesIO


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
    image.save(buffer, format=config.IMAGE_FORMAT)
    return base64.b64encode(buffer.getvalue()).decode()


def infer_single_image(image, prompt=None):
    """单张图片推理"""
    if prompt is None:
        prompt = config.DEFAULT_PROMPT
    
    # 转换为 Base64
    img_base64 = image_to_base64(image)
    
    # 初始化客户端
    client = OpenAI(
        api_key="dummy",
        base_url=config.MODEL_API_BASE,
        timeout=120.0
    )
    
    # 调用 API
    response = client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P
    )
    
    return response.choices[0].message.content


def process_document(file_path, progress=None):
    """
    处理文档
    
    ✅ 关键修复：使用 'is not None' 而不是直接 if progress
    """
    file_path = Path(file_path)
    
    # 判断文件类型
    if file_path.suffix.lower() == '.pdf':
        if progress is not None:  # ✅ 修复点
            progress(0, desc="Converting PDF...")
        images = pdf_to_images(file_path)
    else:
        images = [Image.open(file_path)]
    
    # 逐页推理
    results = []
    total = len(images)
    
    for i, img in enumerate(images):
        if progress is not None:  # ✅ 修复点
            progress((i + 1) / total, desc=f"Processing page {i + 1}/{total}...")
        
        result = infer_single_image(img)
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