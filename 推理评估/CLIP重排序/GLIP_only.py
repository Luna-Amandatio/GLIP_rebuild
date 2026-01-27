import os


# 补丁
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


import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import json
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


#常量定义
num = 500
confidence = 0.3
List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book',
 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table',
 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


#数据集基础目录
base_path = "D:/Project/ComSen/GLIP/DATASET/Coco.v1i.coco-segmentation_2"
json_path = base_path + "/valid/_annotations.coco.json"
images_path = base_path + "/valid/"

# GLIP配置
cfg_file = r"D:\Project\ComSen\GLIP\configs\pretrain\glip_Swin_T_O365_GoldG.yaml"
weight_file = r"D:\Project\ComSen\GLIP\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(cfg_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

#实例初始化
glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=confidence)


def run_success_guaranteed_map(List):
    #HIGH_SUCCESS_CLASSES = ["person", "car", "dog", "bicycle", "cat"]

    #提示词
    caption = List

    # COCO评估对象
    coco_gt = COCO(json_path)
    predictions = []
    success_count = 0

    # 类别映射到ID
    category_map = {}
    for class_name in List:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])
        category_map[class_name] = cat_ids[0]
    print(category_map)

    # 获取图像ID，用集合去重
    all_img_ids = set()
    for class_name in List:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])

        img_ids = coco_gt.getImgIds(catIds=cat_ids)
        all_img_ids.update(img_ids)

    img_ids = list(all_img_ids)[:num]
    img_infos = coco_gt.loadImgs(img_ids)

    for i, img_info in enumerate(img_infos):
        img_path = images_path + img_info['file_name']
        image = np.array(Image.open(img_path).convert('RGB'))

        print(f"[{i + 1}/{len(img_infos)}] {img_info['file_name']})")

        preds = glip_demo.compute_prediction(image, caption)
        top_preds = glip_demo._post_process(preds, threshold=confidence)

        labels = top_preds.get_field("labels").tolist()
        scores = top_preds.get_field("scores").tolist()
        print(scores)
        boxes = top_preds.bbox.detach().cpu().numpy()


        #保存预测结果
        for box, score, label_id in zip(boxes, scores, labels):
            pred_idx = label_id - 1  # GLIP标签从1开始
            pred = {
                "image_id": img_info['id'],
                "category_id": int(pred_idx),
                "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                "score": float(score)
            }
            predictions.append(pred)

        if len(labels) > 0:
            success_count += 1
            print(f"检测到 {len(labels)} 个目标")
        else:
            print(f"未检测到目标")

    #保存预测结果
    pred_file = "glip_only_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    # mAP计算
    if predictions:
        cocoDt = coco_gt.loadRes(pred_file)
        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")

        eval_cat_ids = list(category_map.values())
        cocoEval.params.catIds = eval_cat_ids

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print(f"GLIP mAP计算结果:{cocoEval.stats[1]*100:.5f}%")
        with open('glip_only.txt', 'w', encoding='utf-8') as f:
            f.write(f'mAP结果：{cocoEval.stats[1]*100:.7f}\n')
    else:
        print("无预测结果")


if __name__ == '__main__':
    run_success_guaranteed_map(List)
