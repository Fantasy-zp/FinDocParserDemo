"""
FinDocParser 配置文件 - Phase 2
"""

# ============================================
# 多模型配置
# ============================================
MODELS = {
    "qwen2_5vl_finetuned": {
        "name": "FinDocParserV1 ⭐",
        "api_base": "http://localhost:8001/v1",
        "model_id": "/data/cby/联合训练/chectpoint/第二版数据/内容+跨页/15epoch/v1-20251105-145645/checkpoint-500-merged",
        "type": "openai",  # ✅ 添加类型标识
        "description": "微调后的模型，专为金融文档优化",
        "max_tokens": 8192,
        "recommended": True
    },
    "qwen2_5vl_base": {
        "name": "Qwen2.5-VL-3B (Base)",
        "api_base": "http://localhost:8002/v1",
        "model_id": "Qwen2.5-VL-3B",
        "type": "openai",  # ✅ 添加类型标识
        "description": "基础模型",
        "max_tokens": 8192,
        "recommended": False
    },

    # ============================================
    # ✅ 新增：跨页合并模型
    # ============================================
    "cross_page_fusion": {
        "name": "跨页合并模型 🔗",
        "api_base": "http://127.0.0.1:8002",
        "model_id": "cross-page-fusion",
        "type": "custom",  # ✅ 自定义类型
        "description": "支持跨页表格自动合并，专为复杂金融文档设计",
        "max_tokens": None,  # 不支持
        "supports_params": False,  # ✅ 标记不支持参数调整
        "supports_prompt": False,   # ✅ 标记不支持自定义 prompt
        "recommended": True
    }
}

# 默认模型
DEFAULT_MODEL = "qwen2_5vl_finetuned"   # 或改为 "cross_page_fusion"

# ============================================
# 推理参数（可调节的默认值）
# ============================================
DEFAULT_TEMPERATURE = 0.0001
DEFAULT_TOP_P = 0.3
DEFAULT_MAX_TOKENS = 8192

# ============================================
# Prompt 模板
# ============================================
DEFAULT_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "ALL tables should be presented in HTML format.\n"
    "If there are images or figures in the page, present them as \"<Image>(left,top),(right,bottom)</Image>\", "
    "(left,top,right,bottom) are the coordinates of the top-left and bottom-right corners of the image or figure.\n"
    "Present all titles and headings as H1 headings.\n"
    "Do not hallucinate.\n"
)

# ============================================
# PDF 处理配置
# ============================================
PDF_DPI = 200
IMAGE_FORMAT = "PNG"

# ============================================
# 图像处理配置
# ============================================
IMAGE_MAX_PIXELS = 589824  # 768x768
IMAGE_MIN_PIXELS = 1024    # 32x32

# ============================================
# 界面配置
# ============================================
TITLE = "端到端金融文档解析——FinDocParserV1"
DESCRIPTION = """
支持上传金融文档（PDF 或图片）并转换为 Markdown 格式。
"""

# ============================================
# Examples 配置
# ============================================
EXAMPLES_DIR = "examples"  # 示例文件目录

# ============================================
# 文件配置
# ============================================
ALLOWED_FILE_TYPES = [".pdf", ".png", ".jpg", ".jpeg"]
MAX_FILE_SIZE_MB = 50

# ============================================
# Phase 3.1: 并行处理配置
# ============================================
PARALLEL_ENABLED = True  # 是否启用并行
MAX_WORKERS = 4          # 最大并发线程数（推荐 2-8）
PARALLEL_MIN_PAGES = 2   # 少于此页数不启用并行（单页直接处理）


# ============================================
# Phase 3.3: 缓存配置
# ============================================
CACHE_ENABLED = True            # 是否启用缓存
CACHE_DIR = "cache"             # 缓存目录
CACHE_MEMORY_SIZE = 100         # 内存缓存容量（最多缓存100个文档）
CACHE_DISK_SIZE_MB = 1000       # 磁盘缓存大小限制（1GB）
CACHE_TTL_DAYS = 7              # 缓存有效期（7天）