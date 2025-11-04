# test_api.py
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8002/v1"
)

# 测试 1：列出模型
try:
    models = client.models.list()
    print("✅ API 连接成功")
    print(f"可用模型: {[m.id for m in models.data]}")
except Exception as e:
    print(f"❌ API 连接失败: {e}")
    exit(1)

# 测试 2：简单推理（纯文本）
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=50
    )
    print(f"✅ 推理成功: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ 推理失败: {e}")