# FinDocParserDemo

端到端金融文档解析 Demo。上传 PDF 或图片后，调用视觉语言模型（VLM）将文档转换为 Markdown，并在页面中同时展示原始文档、解析预览、版面分析图和原始输出，支持流式进度、结果下载与本地缓存。

## 功能特性

- 支持多页 PDF 及 PNG / JPG / JPEG 图片，单文件最大 50 MB
- 支持两类模型接入：
  - `custom`：OCRFlux 风格文档解析 API，整份文档上传，支持跨页内容合并与版面图返回
  - `openai`：OpenAI 兼容的 VLM 接口，逐页并发推理
- 流式展示解析进度、已用时和预计剩余时间
- 右侧结果区分四个标签页：原始文档、预览、版面分析、源码
- 支持 Markdown 下载
- 上传后自动生成文档预览，便于与解析结果对照
- 两级缓存（内存 LRU + SQLite 磁盘），重复文档可秒级返回
- 内置示例文档区域，目录中存在文件时自动生成预览
- 服务绑定 `0.0.0.0`，支持局域网内其他设备访问

## 处理流程

```text
上传 PDF / 图片
      |
      +-- PDF  ->  PyMuPDF 按页转图（默认 200 DPI）
      |
      +-- custom 模型  ->  POST {api_base}/parse（整份文件）
      |                     -> document_text + 版面图
      |
      +-- openai 模型  ->  逐页调用 chat.completions（图片 base64）
                            -> 按页顺序拼接 Markdown
      |
      +-- Markdown 预览 / 源码 / 下载
      |
      +-- 有效结果写入缓存（内存 + SQLite）
```

## 目录结构

```text
FinDocParserDemo/
├── app.py               # Gradio 界面、事件绑定、流式结果处理
├── utils.py             # PDF 转图、模型调用、缓存集成、版面图处理
├── config.py            # 模型列表、推理参数、文件与缓存配置
├── cache_manager.py     # 两级缓存实现（LRU + SQLite）
├── assets/              # Logo 等静态资源
├── examples/            # 示例文档目录（默认被 .gitignore 忽略）
├── test/                # 独立测试脚本
│   ├── test_api.py
│   ├── test_cache.py
│   ├── test_image_api.py
│   ├── test_parallel.py
│   └── test_streaming.py
├── cache/               # 运行时缓存目录（自动创建，默认忽略）
└── README.md
```

## 环境要求

- Python 3.10+
- 可访问的模型服务（本地或局域网）
- 已验证环境：Python 3.10.18、Gradio 5.44.1、PyMuPDF 1.26.5、Pillow 11.1.0、openai 1.97.1、requests 2.31.0、cachetools 6.2.4

主要依赖：

```bash
pip install gradio==5.44.1 PyMuPDF==1.26.5 Pillow==11.1.0 openai==1.97.1 requests==2.31.0 cachetools==6.2.4
```

## 快速开始

```bash
git clone https://github.com/Fantasy-zp/FinDocParserDemo.git
cd FinDocParserDemo

# 建议使用独立环境，例如：
conda create -n findoc python=3.10 -y
conda activate findoc
pip install gradio==5.44.1 PyMuPDF==1.26.5 Pillow==11.1.0 openai==1.97.1 requests==2.31.0 cachetools==6.2.4

python app.py
```

启动后访问：

- 本机：<http://localhost:7860>
- 局域网：`http://<本机局域网IP>:7860`

获取本机局域网 IP（macOS）：

```bash
ipconfig getifaddr en0
```

服务启动参数位于 [app.py](app.py) 末尾：

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
    favicon_path="assets/logo.ico"
)
```

## 模型配置

模型列表在 [config.py](config.py) 的 `MODELS` 中定义，`DEFAULT_MODEL` 指定默认模型。

| 显示名称 | 类型 | 接口地址 | 说明 |
| --- | --- | --- | --- |
| FinDocParserV2 | `custom` | `http://192.168.100.230:8003` | 支持跨页表格合并与版面分析，默认模型 |
| FinDocParserV1.5 | `custom` | `http://127.0.0.1:8003` | 具备元素内容解析能力 |
| FinDocParserV1 | `openai` | `http://localhost:8001/v1` | 微调后的 Qwen2.5-VL，逐页推理 |
| Qwen2.5-VL-3B (Base) | `openai` | `http://localhost:8002/v1` | 基础模型 |

两类模型的字段差异：

- `custom` 模型需要 `api_base`，请求 `POST {api_base}/parse` 和 `GET {api_base}/health`
- `openai` 模型需要 `api_base`（含 `/v1`）和 `model_id`，通过 `models.list()` 与 `chat.completions.create()` 调用

推理参数在页面「高级设置」中调整，默认值来自 `config.py`：

- `temperature`：0.0001
- `top_p`：0.3
- `max_tokens`：8192
- 自定义提示词：留空时使用 `config.DEFAULT_PROMPT`

`custom` 模型不支持提示词和采样参数，页面中的相关设置对其无效。

## 后端接口约定

### custom 模型

```http
POST {api_base}/parse
Content-Type: multipart/form-data

file=<PDF 或图片>
```

期望返回：

```json
{
  "success": true,
  "result": {
    "num_pages": 1,
    "document_text": "# Markdown 内容...",
    "page_texts": {},
    "output_imgs": ["<base64 PNG>"]
  },
  "timing": {}
}
```

- 正文取自 `result.document_text`
- 版面图优先取顶层 `layout_images`，兼容新版 `result.output_imgs`，均为 base64 图片列表
- 健康检查：`GET {api_base}/health`，返回 `{"status": "ok"}` 视为可用

### openai 兼容模型

```http
POST {api_base}/chat/completions
```

页面会将每一页转成 JPEG base64 后作为 `image_url` 传入，按页返回 Markdown。多页时使用线程池并发（`config.MAX_WORKERS`，默认 4）。

## 界面说明

左侧输入区：

- 上传 PDF 或图片，上传后自动预览
- 选择模型并测试连接
- 高级设置：温度、Top P、最大生成长度、自定义提示词
- 缓存管理：查看统计、清空缓存
- 点击「开始解析」触发流式解析

右侧结果区：

- 「原始文档」：上传文件逐页预览
- 「预览」：渲染后的 Markdown
- 「版面分析」：`custom` 模型返回的版面可视化图
- 「源码」：原始 Markdown / 后端原始 JSON

## 缓存机制

实现位于 [cache_manager.py](cache_manager.py)，默认配置在 [config.py](config.py)：

- 内存缓存：LRU，最多 100 条
- 磁盘缓存：SQLite（`cache/cache.db`），上限 1 GB，默认有效期 7 天
- 缓存键：
  - `custom`：`SHA-256(文件内容)[:16] + model_key`
  - `openai`：`SHA-256(文件内容)[:16] + MD5(model_key + prompt + temperature + top_p + max_tokens)[:16]`
- 仅缓存校验通过的结果；解析失败不会写入缓存
- 超过磁盘上限时按访问次数和访问时间淘汰约 20% 的记录

注意：

- `custom` 模型的缓存键不包含提示词和采样参数，修改这些参数后仍会命中旧缓存
- 旧版本缓存可能不包含版面图，升级后如版面分析为空，可在页面中清空缓存或重新解析

## 测试

测试脚本为独立运行脚本，不会自动发现，需在项目根目录手动执行：

```bash
python test/test_api.py
python test/test_image_api.py
python test/test_streaming.py
python test/test_parallel.py
python test/test_cache.py
```

| 脚本 | 依赖 |
| --- | --- |
| `test_api.py` | `localhost:8002/v1` 服务 |
| `test_image_api.py` | `localhost:8002/v1` 服务，支持图片推理 |
| `test_streaming.py` | `examples/1223836236.pdf`，`localhost:8001/v1` 服务 |
| `test_parallel.py` | `examples/1223836236.pdf`，`localhost:8001/v1` 服务 |
| `test_cache.py` | `examples/1223836236.pdf`，`localhost:8001/v1` 服务 |

`examples/` 和 `assets/logo.png` 默认被 `.gitignore` 忽略，克隆仓库后需要自行放入示例文档；缺少 Logo 时页面会回退为文字标识，不影响运行。

## 常见问题

**启动时报 `ModuleNotFoundError: No module named 'gradio'`**

使用了未安装依赖的 Python 环境。先激活正确的环境，再执行 `python app.py`，可用 `which python` 确认解释器路径。

**局域网其他设备打不开**

1. 确认服务监听在 `0.0.0.0:7860`，可用 `lsof -nP -iTCP:7860 -sTCP:LISTEN` 检查
2. macOS 防火墙是否允许 Python 接收入站连接
3. 路由器是否开启了 AP 隔离
4. 确认访问方与本机在同一网段

**版面分析标签页为空**

1. 确认当前使用的是 `custom` 模型（`openai` 模型不返回版面图）
2. 确认后端响应包含 `result.output_imgs` 或顶层 `layout_images`，且为 base64 图片
3. 清空旧缓存后重新解析，避免命中不含版面图的历史缓存

**模型连接测试失败**

检查模型服务是否启动、`api_base` 是否可达、端口是否正确、`openai` 类型的 `model_id` 是否存在。

**示例文档区域为空**

`examples/` 目录不存在或没有受支持的文件，或被 `.gitignore` 忽略。放入 PDF / 图片后刷新页面即可。

## 说明与限制

- 本项目为演示用途，未实现用户认证与权限控制，请仅在可信局域网内使用，不要直接暴露到公网
- 上传文件与解析结果保存在本机，局域网用户共用同一服务实例
- `custom` 模型为整份文档一次性请求，超时时间默认 300 秒，大文档解析耗时较长
- `openai` 模型按页推理，跨页表格与跨页语义不会自动合并
