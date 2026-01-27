from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import re
import itertools

#blip模型路径
model_path = "D:/model/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to("cuda")

# 加载本地图像
raw_image = Image.open("../tennis.jpg").convert('RGB')

# 生成描述
inputs = processor(raw_image, return_tensors="pt").to("cuda")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("BLIP描述:", caption)

colors = 'Red|Orange|Yellow|Green|Blue|Purple|Pink|Black|White|Gray|Brown|Gold|Silver'

#提取冠词后名词
def extract_1(sentence):
    # 正则匹配 "a " 或 "an " 后的连续单词（匹配到下一个空格或标点为止）
    pattern = fr'\b(a|an)\s+(\w+)'
    matches = re.findall(pattern, sentence, re.IGNORECASE)  # 忽略大小写
    # 返回所有匹配到的后的单词"
    return [word for (article, word) in matches]

#提取颜色词后名词
def extract_2(sentence):
    pattern = fr'\b({colors})\s+(\w+)'
    matches = re.findall(pattern, sentence, re.IGNORECASE)  # 忽略大小写
    combined_list = [f"{prefix} {word}" for (prefix, word) in matches]
    return combined_list

#提取ing动词后名词
def extract_3(sentence):
    pattern = r'\b\w+ing\s+(\w+)\b'
    matches = re.findall(pattern, sentence, re.IGNORECASE)
    return matches

list_1 = extract_1(caption)
list_2 = extract_2(caption)
list_3 = extract_3(caption)

combined_list = list(itertools.chain(list_1, list_2,list_3))
result = '.'.join(combined_list)