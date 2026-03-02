'''
多提示词推理结果
'''

import warnings
warnings.filterwarnings("ignore")

#-------------------------------
#保证本地环境的中文分词器得以加载，避免联网检查
#-------------------------------
import os
import nltk

def no_download(*args, **kwargs):
    """禁用所有下载尝试"""
    return None


nltk.download = no_download
nltk.data.url = lambda x: None  # 禁用URL访问
# 强制本地路径
nltk.data.path = [r'C:\Users\LQY1\AppData\Roaming\nltk_data']

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NLTK_QUIET'] = 'True'


import cv2
import numpy as np
import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from PIL import Image, ImageDraw, ImageFont
from transformers import logging
logging.set_verbosity_error()

IMAGE_path = './example/tennis.jpg'
'''
'tennis',
    'a man.a racket.a tennis',
    'a photo of a man.a photo of a racket.a photo of a tennis',
    'a black man.white and black racket.white tennis',
    'a black man.white and black racket.spherical white tennis'

    '
'''
prompts = [
    'a tennis',
    'a photo of a tennis',
    'white tennis',
    'spherical white tennis'
]


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        hexs = (
            "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
            "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def draw_images(image, boxes, classes, scores, colors, xyxy=True):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[:, :, ::-1])
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    # 设置字体,pillow 绘图环节
    font = ImageFont.truetype(font="C:/Windows/Fonts/arial.ttf",
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    # 多次画框的次数,根据图片尺寸不同,把框画粗
    thickness = max((image.size[0] + image.size[1]) // 300, 1)
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i]

        label = '{}:{:.2f}'.format(classes[i], scores[i])
        tx1, ty1, tx2, ty2 = font.getbbox(label)
        tw, th = tx2 - tx1, ty2 - tx1

        text_origin = np.array([x1, y1 - th]) if y1 - th >= 0 else np.array([x1, y1 + 1])

        # 在目标框周围偏移几个像素多画几次, 让边框变粗
        for j in range(thickness):
            draw.rectangle((x1 + j, y1 + j, x2 - j, y2 - j), outline=color)
        # 画标签
        draw.rectangle((text_origin[0], text_origin[1], text_origin[0] + tw, text_origin[1] + th), fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    return image


config_file = r"..\..\configs\pretrain\glip_Swin_T_O365_GoldG.yaml"
weight_file = r'..\..\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth'

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.3,
    show_mask_heatmaps=False
)


def glip_inference(image_, caption_):
    # 为不同类别设置颜色, 从caption提取的类别不同
    colors_ = Colors()

    preds = glip_demo.compute_prediction(image_, caption_)
    top_preds = glip_demo._post_process(preds, threshold=0.4)

    # 从预测结果中提取预测类别,得分和检测框
    labels = top_preds.get_field("labels").tolist()
    scores = top_preds.get_field("scores").tolist()
    boxes = top_preds.bbox.detach().cpu().numpy()

    # 映射回文字
    phrases = caption_.split('.')
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    # 为每个预测类别设置框颜色
    colors = [colors_(idx) for idx in labels]

    IDX = []
    labels_names = []
    for i, label in enumerate(labels):
        phrase_idx = int(label) - 1
        if 0 <= phrase_idx < len(phrases):  # 确保索引有效
            IDX.append(phrase_idx)
    # 获得标签数字对应的类别名
    scores = [scores[i] for i in IDX]
    for i in IDX:
        labels_names.append(phrases[i])

    boxes = boxes[IDX]

    return boxes, scores, labels_names, colors


if __name__ == '__main__':
    # caption = 'bobble heads on top of the shelf'
    # caption = "Striped bed, white sofa, TV, carpet, person"
    # caption = "table on carpet"
    for caption in prompts:
        image = cv2.imread(IMAGE_path)
        boxes, scores, labels_names, colors = glip_inference(image, caption)
        print(labels_names, scores)
        print(boxes)
        image = draw_images(image=image, boxes=boxes, classes=labels_names, scores=scores, colors=colors)
        image.save(f'./{caption}.jpg')

