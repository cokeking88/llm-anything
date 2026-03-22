# 06_多模态LLM

## 什么是多模态LLM？

**一句话**：能同时理解文字、图片、声音、视频的AI

**类比**：
```
普通LLM：只能看文字
多模态LLM：像人一样，能看图、听声音、读文字

GPT-4o：能看图片、听语音、生成图片
```

---

## 主流多模态模型

| 模型 | 公司 | 支持模态 | 特点 |
|------|------|----------|------|
| GPT-4o | OpenAI | 文字+图片+语音 | 最强多模态 |
| Gemini 1.5 | Google | 文字+图片+视频+音频 | 百万token |
| Claude 3.5 | Anthropic | 文字+图片 | 长上下文 |
| LLaVA | 开源 | 文字+图片 | 开源可部署 |
| Qwen-VL | 阿里 | 文字+图片 | 中文优秀 |

---

## 多模态原理

### 架构

```
图片 → 图像编码器 → 图像Token
                              ↓
文字 → 文字编码器 → 文字Token → Transformer → 输出
音频 → 音频编码器 → 音频Token
```

### 图像理解

```
图片 → Vision Encoder → 图像特征向量
                              ↓
                         投影层
                              ↓
                        LLM处理
                              ↓
                         文字描述
```

---

## 使用多模态模型

### GPT-4V图片理解

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### 使用开源模型LLaVA

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

# 加载模型
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# 加载图片
image = Image.open("image.jpg")

# 生成描述
prompt = "[INST] <image>\n描述这张图片 [/INST]"
inputs = processor(prompt, image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 多模态应用

### 1. 图片问答

```
用户：[图片] 这张图片里有什么？
AI：图片中有一只金毛犬在草地上奔跑，背景是蓝天白云。
```

### 2. OCR文字识别

```
用户：[图片] 提取图片中的文字
AI：图片中的文字是："欢迎来到人工智能世界"
```

### 3. 图片生成

```
用户：画一只可爱的猫
AI：[生成图片]
```

### 4. 视频理解

```
用户：[视频] 总结这个视频的内容
AI：这个视频展示了制作蛋糕的全过程，包括准备材料、搅拌、烘烤等步骤。
```

---

## 语音多模态

### Whisper语音识别

```python
import whisper

# 加载模型
model = whisper.load_model("base")

# 识别语音
result = model.transcribe("audio.mp3")
print(result["text"])
```

### GPT-4o语音对话

```python
# GPT-4o支持实时语音对话
# 可以直接用语音提问，AI用语音回答
```

---

## 开源多模态模型

### LLaVA（图片理解）

```bash
# 安装
pip install llava

# 使用
from llava.model import LlavaLlamaForCausalLM

model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")
```

### Qwen-VL（中文多模态）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 图片问答
response = model.chat(tokenizer, "描述这张图片", image="image.jpg")
```

### Stable Diffusion（图片生成）

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("a cute cat").images[0]
image.save("cat.png")
```

---

## 多模态RAG

### 图片+文字RAG

```python
# 1. 提取图片中的文字（OCR）
# 2. 将文字和图片描述存入向量库
# 3. 检索时同时搜索文字和图片
```

### 实现

```python
from langchain.document_loaders import ImageCaptionLoader

# 加载图片并生成描述
loader = ImageCaptionLoader(["image1.jpg", "image2.jpg"])
documents = loader.load()

# 存入向量库
vectorstore = FAISS.from_documents(documents, embeddings)
```

---

## 学习资源
- [GPT-4V文档](https://platform.openai.com/docs/guides/vision)
- [LLaVA论文](https://arxiv.org/abs/2304.08485)
- [Hugging Face多模态模型](https://huggingface.co/models?pipeline_tag=image-text-to-text)

## 下一步
学习 [[07_前沿技术]]
