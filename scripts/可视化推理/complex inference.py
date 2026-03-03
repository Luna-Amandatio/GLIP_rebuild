'''
多提示词推理结果 - 仅GLIP版本
'''

import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 保证本地环境的中文分词器得以加载，避免联网检查
# -------------------------------
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
from torchvision.ops import nms

IMAGE_path = './example/tennis.jpg'


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


class GLIPOnly:
    """
    仅使用GLIP进行推理的简化版本
    """

    def __init__(self,
                 caption,
                 image_path,
                 confidence=0.1,
                 iou_threshold=0.5,
                 nms=False):

        self.nms = nms
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # 处理caption
        if isinstance(caption, str):
            self.caption = caption.split('.')
            self.caption = [phrase.strip() for phrase in self.caption if phrase.strip()]
        else:
            self.caption = caption

        # 加载GLIP模型
        self.glip = self.glip_load()
        self.image_path = image_path

        self.labels = []
        self.scores = []
        self.boxes = []
        self.color = None

    def glip_load(self):
        """加载GLIP模型"""
        cfg_file = r"..\..\configs\pretrain\glip_Swin_T_O365_GoldG.yaml"
        weight_file = r"..\..\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth"

        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(cfg_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", self.device])

        glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=self.confidence)
        glip_demo.model.eval()

        return glip_demo

    def model_inference(self):
        """仅GLIP模型推理"""
        image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)

        if image is None:
            print(f"错误：无法加载图片 {self.image_path}")
            return

        # GLIP推理
        with torch.no_grad():
            # 将caption列表转换为GLIP需要的格式（用.连接）

            preds = self.glip.compute_prediction(image,self.caption)
            top_preds = self.glip._post_process(preds, threshold=self.confidence)

            if len(top_preds) == 0:
                print("未检测到任何目标")
                return

            # GLIP结果提取
            labels = top_preds.get_field("labels").tolist()
            glip_scores = top_preds.get_field("scores").tolist()
            boxes = top_preds.bbox.detach().cpu().numpy()

            print(f"原始检测到 {len(labels)} 个目标")
            print(f"原始labels: {labels}")
            print(f"原始scores: {[f'{s:.3f}' for s in glip_scores]}")

            # 将labels映射回类别名称（GLIP的labels从1开始）
            temp_predictions = []
            for i, (label, score, box) in enumerate(zip(labels, glip_scores, boxes)):
                phrase_idx = int(label) - 1  # 转换为0开始的索引
                if 0 <= phrase_idx < len(self.caption):
                    class_name = self.caption[phrase_idx]

                    self.labels.append(class_name)
                    self.scores.append(score)
                    self.boxes.append(box)

                    # 为NMS准备数据
                    x1, y1, x2, y2 = box
                    temp_predictions.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': score,
                        'class_name': class_name
                    })

                    print(f"  框{i}: {class_name}, 分数={score:.3f}")

            print(f"有效检测结果: {len(self.labels)} 个目标")

        # NMS后处理
        if self.nms and temp_predictions:
            print(f"\nNMS处理前有 {len(temp_predictions)} 个框")
            nms_results = self.apply_nms(temp_predictions)
            print(f"NMS处理后剩 {len(nms_results)} 个框")

            # 更新结果为NMS后的
            self.boxes = []
            self.scores = []
            self.labels = []
            for result in nms_results:
                x1, y1, w, h = result['bbox']
                self.boxes.append([x1, y1, x1 + w, y1 + h])
                self.scores.append(result['score'])
                self.labels.append(result['class_name'])

        # 生成颜色
        self.color = self.color_chose()

    def apply_nms(self, predictions, conf_threshold=0.05):
        """应用NMS"""
        if not predictions:
            return []

        boxes = []
        scores = []
        for pred in predictions:
            x1, y1, w, h = pred['bbox']
            boxes.append([x1, y1, x1 + w, y1 + h])
            scores.append(pred['score'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        keep_indices = nms(boxes, scores, self.iou_threshold)
        valid_keep = keep_indices[scores[keep_indices] > conf_threshold]

        return [predictions[i.item()] for i in valid_keep]

    def color_chose(self):
        colors_ = Colors()
        return [colors_(i) for i in range(len(self.labels))]


if __name__ == '__main__':
    IMAGE_path = './example/tennis.jpg'


    caption = 'a black man.a white racket.spherical white tennis.a black woman.a baseball.a dog.a cat.a car.a bicycle'


    # 创建GLIP推理实例
    glip_only = GLIPOnly(
        caption=caption,
        image_path=IMAGE_path,
        confidence=0.1,  # GLIP置信度阈值
        iou_threshold=0.1,  # NMS的IoU阈值
        nms=True  # 是否使用NMS
    )

    # 执行推理
    glip_only.model_inference()

    # 显示结果
    if len(glip_only.boxes) > 0:
        image = cv2.imread(IMAGE_path)
        image = draw_images(
            image=image,
            boxes=glip_only.boxes,
            classes=glip_only.labels,
            scores=glip_only.scores,
            colors=glip_only.color
        )

        # 保存结果
        image.save("未进行CLIP重排结果.jpg")
        image.show()
    else:
        print("没有检测到任何目标")