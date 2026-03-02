'''
用于使用blip模型获取提示词 - 带智能去重
'''
# -------------------------------
# 保证本地环境的中文分词器得以加载，避免联网检查
# -------------------------------
import nltk

def no_download(*args, **kwargs):
    """禁用所有下载尝试"""
    return None

nltk.download = no_download
nltk.data.url = lambda x: None  # 禁用URL访问
# 强制本地路径
nltk.data.path = [r'C:\Users\LQY1\AppData\Roaming\nltk_data']


import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NLTK_QUIET'] = 'True'

# -------------------------------
# 导入 NLTK 所需模块
# -------------------------------
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

import torch
import warnings
warnings.filterwarnings('ignore')


class KeywordGeneration:
    '''
    该类负责使用blip与分词器生成glip格式提示词

    Attributes:
        img:输入的图片
        device:推理设备
        caption:blip生成的描述语句
        noun_phrases:分割生成的提示词列表
        result:最终生成的glip提示词
    '''
    def __init__(self,img):
        self.img = img
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.caption = self.blip_load()
        self.noun_phrases=self.extract_noun_phrases_from_pos()
        self.result = '.'.join(self.noun_phrases)


    def blip_load(self):
        # blip模型路径
        model_path = "../MODEL/blip-image-captioning-base"
        model_blip = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)
        processor_blip = BlipProcessor.from_pretrained(model_path, local_files_only=True,use_fast=True)

        #短语提取# 生成描述
        inputs_blip = processor_blip(raw_image, return_tensors="pt").to("cuda")
        out_blip = model_blip.generate(**inputs_blip)
        caption = processor_blip.decode(out_blip[0], skip_special_tokens=True)
        print("BLIP描述:", caption)
        return caption

    def extract_noun_phrases_from_pos(self):
        """
        仅基于词性标注提取名词短语
        返回: 完整名词短语列表
        """
        #分词
        words = word_tokenize(self.caption)
        #词性标注
        pos_tags = pos_tag(words)
        # 名词短语的构成部分词性
        # DT: 冠词 (a, an, the)
        # JJ: 形容词
        # NN: 名词单数
        # NNS: 名词复数
        # NNP: 专有名词单数
        # NNPS: 专有名词复数
        noun_phrase_tags = ['DT', 'JJ', 'NN', 'NNS', 'NNP', 'NNPS']

        noun_phrases = []  # 完整名词短语

        i = 0
        while i < len(pos_tags):
            word, tag = pos_tags[i]

            # 如果当前词是名词短语的开始（冠词、形容词或名词）
            if tag in noun_phrase_tags:
                phrase_words = [word]

                # 继续向后收集名词短语的组成部分
                j = i + 1
                while j < len(pos_tags):
                    next_word, next_tag = pos_tags[j]

                    # 如果下一个词也是名词短语的一部分，继续收集
                    if next_tag in noun_phrase_tags:
                        phrase_words.append(next_word)
                        j += 1
                    else:
                        break

                # 提取完整名词短语
                full_phrase = ' '.join(phrase_words)
                noun_phrases.append(full_phrase)

                # 跳过已处理的词
                i = j
            else:
                i += 1

        return noun_phrases

    def clip_rerank(self):
        # CLIP加载
        model_clip = CLIPModel.from_pretrained(r"D:\model\CLIP").to(self.device)
        processor_clip = CLIPProcessor.from_pretrained(r"D:\model\CLIP", use_fast=True)
        # CLIP 筛选
        if self.noun_phrases:
            print("\n" + "=" * 50)
            print("CLIP 相似度筛选:")
            print("=" * 50)

            # 用 CLIP 计算每个关键词的相似度
            inputs_clip = processor_clip(
                text=self.noun_phrases,  # 先用智能去重后的列表
                images=raw_image,
                return_tensors="pt",
                padding=True
            )

            outputs = model_clip(**inputs_clip)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]

            # 创建带分数的列表
            keyword_scores = list(zip(self.noun_phrases, probs.tolist()))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)

            print("\n关键词相似度评分（从高到低）:")
            for keyword, prob in keyword_scores:
                print(f"  {keyword}: {prob:.4f}")

#---------------------------------
#测试
# ---------------------------------
if __name__ == "__main__":
    # 加载本地图像
    raw_image = Image.open("../1.jpg").convert('RGB')
    #类实例化
    keyword = KeywordGeneration(raw_image)
    #生成glip格式关键词
    print("\n获取的提示词:", keyword.result)
