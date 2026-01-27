from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载模型和处理器
model = CLIPModel.from_pretrained(r"D:\model\CLIP")
processor = CLIPProcessor.from_pretrained(r"D:\model\CLIP",use_fast=True)

image = Image.open("bear.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bear"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图像与文本的相似度
probs = logits_per_image.softmax(dim=1).tolist()[0] # 转换为概率

print(probs)

