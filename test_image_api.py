"""
测试图像推理
"""
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

# 创建一个简单的测试图像
img = Image.new('RGB', (400, 100), color='white')
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(img)
draw.text((10, 40), "Test Document", fill='black')

# 转换为 Base64
buffer = BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode()

# 调用 API
client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8001/v1",
    timeout=60.0
)

print("🧪 Testing image inference...")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用 API 返回的模型 ID
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
                    "text": "请描述这张图片的内容"
                }
            ]
        }],
        max_tokens=100
    )
    
    print(f"✅ 推理成功!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()