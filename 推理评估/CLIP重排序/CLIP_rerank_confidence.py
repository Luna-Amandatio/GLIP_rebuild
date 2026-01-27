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
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision.ops import nms

#常量定义
num = 30
#bed,laptop
List = ['bed','laptop']
'''
List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book',
 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table',
 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
'''



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


#CLIP加载
model = CLIPModel.from_pretrained(r"D:\model\CLIP")
processor = CLIPProcessor.from_pretrained(r"D:\model\CLIP",use_fast=True)


def apply_nms_post_fusion(temp_predictions, iou_threshold=0.5, conf_threshold=0.1):
    """使用torchvision.ops.nms对融合预测结果后处理"""
    if not temp_predictions:
        return []

    # 转换为torch tensor格式
    boxes = []
    scores = []
    for pred in temp_predictions:
        # torchvision.nms 需要 [x1,y1,x2,y2] 格式
        x1, y1, w, h = pred['bbox']
        boxes.append([x1, y1, x1 + w, y1 + h])
        scores.append(pred['score'])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # 执行NMS，返回保留的索引
    keep_indices = nms(boxes, scores, iou_threshold)

    # 过滤掉分数太低的框
    valid_keep = keep_indices[scores[keep_indices] > conf_threshold]

    # 转换回原格式
    final_preds = [temp_predictions[i.item()] for i in valid_keep]

    return final_preds

def run_success_guaranteed_map(List,confidence,alpha):
    #HIGH_SUCCESS_CLASSES = ["person", "car", "dog", "bicycle", "cat"]
    glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=confidence)
    #提示词
    caption = List

    # COCO评估对象
    coco_gt = COCO(json_path)
    predictions = []
    success_count = 0

    #类别映射到ID
    category_map = {}
    for class_name in List:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])
        category_map[class_name] = cat_ids[0]

    # 获取图像ID，用集合去重
    all_img_ids = set()
    for class_name in List:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])

        img_ids = coco_gt.getImgIds(catIds=cat_ids)
        all_img_ids.update(img_ids)

    img_ids = list(all_img_ids)[:num]
    img_infos = coco_gt.loadImgs(img_ids)

    for i, img_info in enumerate(img_infos):
        temp_predictions = []
        img_path = images_path + img_info['file_name']
        I = Image.open(img_path)
        image = np.array(I.convert('RGB'))

        print(f"[{i + 1}/{len(img_infos)}] {img_info['file_name']})")

        preds = glip_demo.compute_prediction(image, caption)
        top_preds = glip_demo._post_process(preds, threshold=confidence)

        #labels = top_preds.get_field("labels").tolist()
        glip_score = top_preds.get_field("scores").tolist()
        boxes = top_preds.bbox.detach().cpu().numpy()

        if len(boxes) == 0:
            print("LIP未检测到任何框")
            continue
        print(f"GLIP检测到 {len(boxes)} 个候选框")

        cropped = I.crop(boxes)
        inputs = processor(text=List, images=cropped, return_tensors="pt", padding=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}  # 移到GPU
        with torch.no_grad():
            outputs = model(**inputs)

            probs = outputs.logits_per_image.softmax(dim=1)[0]  # [80个类别的分数]
            best_idx = probs.argmax().item()
            best_class = List[best_idx]
            best_clip_score = probs[best_idx].item()



        #融合分数
        fused_score = glip_score * alpha + best_clip_score * (1-alpha)
        pred = {
            "image_id": img_info['id'],
            "category_id": category_map[best_class],
            "bbox": [float(boxes[0]), float(boxes[1]), float(boxes[2] - boxes[0]), float(boxes[3] - boxes[1])],
            "score": float(fused_score)
        }
        temp_predictions.append(pred)
        nms_results = apply_nms_post_fusion(temp_predictions, iou_threshold=0.5)
        predictions.extend(nms_results)

    #保存预测结果
    pred_file = "glip_clip.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    #mAP计算
    if predictions:
        cocoDt = coco_gt.loadRes(pred_file)
        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")

        eval_cat_ids = list(category_map.values())
        cocoEval.params.catIds = eval_cat_ids

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print(f"GLIP mAP计算结果:{cocoEval.stats[1]*100:.5f}%")
        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write(f'confidence:{confidence},alpha:{alpha},mAP结果：{cocoEval.stats[1]*100:.5f}\n')
    else:
        print("无预测结果")


if __name__ == '__main__':
    for confidence in np.arange(0.0,0.40 , 0.05):
        for alpha in np.arange(0.0,1.0 , 0.1):
            run_success_guaranteed_map(List,confidence,alpha)
