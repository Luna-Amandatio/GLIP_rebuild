'''
比较模型在各置信度下mAP变化
'''
#-------------------------------
#保证本地环境的中文分词器得以加载，避免联网检查
#-------------------------------
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

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import torch

from PIL import Image
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


#常量定义
num = 100
#bed,laptop
CLASSES = 'bird'

#数据集基础目录
base_path = "D:/Project/ComSen/GLIP/DATASET/Coco.v1i.coco-segmentation_2"
json_path = base_path + "/valid/_annotations.coco.json"
images_path = base_path + "/valid/"

# GLIP配置
cfg_file = r"..\..\configs\pretrain\glip_Swin_T_O365_GoldG.yaml"
weight_file = r"..\..\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(cfg_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

#实例初始化



def run_success_guaranteed_map(confidence):
    #HIGH_SUCCESS_CLASSES = ["person", "car", "dog", "bicycle", "cat"]
    glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=confidence)
    #提示词
    caption = CLASSES

    # COCO评估对象
    coco_gt = COCO(json_path)
    predictions = []
    success_count = 0

    # 评估前num张
    CLASSES_ids = coco_gt.getCatIds(catNms=[CLASSES])
    img_ids = coco_gt.getImgIds(catIds=CLASSES_ids)
    img_infos = coco_gt.loadImgs(img_ids)[:num]
    id = CLASSES_ids[0]

    for i, img_info in enumerate(img_infos):
        img_path = images_path + img_info['file_name']
        image = np.array(Image.open(img_path).convert('RGB'))

        print(f"[{i + 1}/{len(img_infos)}] {img_info['file_name']})")

        # 半精度推理
        with torch.cuda.amp.autocast(dtype=torch.float16):
            preds = glip_demo.compute_prediction(image, caption)
            top_preds = glip_demo._post_process(preds, threshold=confidence)

        labels = top_preds.get_field("labels").tolist()
        scores = top_preds.get_field("scores").tolist()
        boxes = top_preds.bbox.detach().cpu().numpy()

        #保存预测结果
        for box, score, label_id in zip(boxes, scores, labels):
            pred = {
                "image_id": img_info['id'],
                "category_id": id,
                "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                "score": float(score)
            }
            predictions.append(pred)

        if len(labels) > 0:
            success_count += 1
            print(f"检测到 {len(labels)} 个目标")
        else:
            print(f"未检测到目标")

    success_rate: float = success_count / num * 100
    print(f"成功率: {success_rate:.5f}%")

    #保存预测结果
    pred_file = "预测结果.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    #mAP计算
    if predictions:
        cocoDt = coco_gt.loadRes(pred_file)
        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")
        cocoEval.params.catIds = CLASSES_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print(f"GLIP mAP计算结果:{cocoEval.stats[1]*100:.5f}%")
        with open('bird.txt', 'a', encoding='utf-8') as f:
            f.write(f'confidence: {confidence} ,mAP结果：{cocoEval.stats[1]*100:.5f}\n')
    else:
        print("无预测结果")


if __name__ == '__main__':
    for confidence in np.arange(0, 1.0, 0.05):
        run_success_guaranteed_map(confidence)
