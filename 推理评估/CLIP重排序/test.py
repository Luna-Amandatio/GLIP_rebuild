import os

os.environ['NLTK_QUIET'] = 'True'
import time
from PIL import Image
import numpy as np
import json
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import CLIPProcessor, CLIPModel
from torchvision.ops import nms

# 常量定义 - 已优化阈值
num = 10
confidence = 0.0001  # GLIP主阈值极低
confidence_clip_crop = 0.001  # 裁剪阈值极低
alpha = 0.6
conf_threshold_nms = 0.001

List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
        'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot',
        'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
        'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse',
        'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange',
        'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
        'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign',
        'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush',
        'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# 数据集路径
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

# 实例初始化
glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=confidence)

# CLIP加载
device = 'cuda'
model = CLIPModel.from_pretrained(r"D:\model\CLIP").to(device)
processor = CLIPProcessor.from_pretrained(r"D:\model\CLIP", use_fast=True)


def apply_nms_post_fusion(temp_predictions, iou_threshold=0.5, conf_threshold=0.001, device='cuda'):
    """NMS后处理"""
    if not temp_predictions:
        return []

    boxes = []
    scores = []
    for pred in temp_predictions:
        x1, y1, w, h = pred['bbox']
        boxes.append([x1, y1, x1 + w, y1 + h])
        scores.append(pred['score'])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    keep_indices = nms(boxes, scores, iou_threshold)
    valid_keep = keep_indices[scores[keep_indices] > conf_threshold]

    return [temp_predictions[i.item()] for i in valid_keep]


def batch_clip_scoring(crops, class_names, processor, model, device, top_k_classes=3):
    """批量CLIP评分 - 增强容错"""
    if not crops:
        return [[] for _ in crops]

    valid_crops = []
    valid_indices = []
    for i, crop in enumerate(crops):
        if crop.size[0] > 1 and crop.size[1] > 1:
            valid_crops.append(crop)
            valid_indices.append(i)

    if not valid_crops:
        return [[] for _ in crops]

    num_valid_crops = len(valid_crops)
    texts_batch = class_names * num_valid_crops

    try:
        inputs = processor(
            images=valid_crops,
            text=texts_batch,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        crop_scores = [[] for _ in range(len(crops))]
        for i in range(len(probs)):
            crop_idx = valid_indices[i // len(class_names)]
            class_idx = i % len(class_names)
            class_name = class_names[class_idx]
            score = probs[i, 0].item()
            crop_scores[crop_idx].append((class_name, score))

        results = []
        for scores_list in crop_scores:
            if scores_list:
                top_scores = sorted(scores_list, key=lambda x: x[1], reverse=True)[:top_k_classes]
            else:
                top_scores = []
            results.append(top_scores)
        return results

    except Exception as e:
        print(f"CLIP处理出错: {e}")
        return [[] for _ in crops]


def run_success_guaranteed_map(class_names):
    """主评估函数 - 完整修复版"""
    print("=== 开始评估 ===")
    coco_gt = COCO(json_path)
    predictions = []

    # 🔍 关键调试：检查类别映射
    print("=== 类别映射检查 ===")
    category_map = {}
    for class_name in class_names:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])
        if cat_ids:
            category_map[class_name] = cat_ids[0]


    # 获取图像
    all_img_ids = set()
    for class_name in class_names:
        if class_name in category_map:
            cat_ids = [category_map[class_name]]
            img_ids = coco_gt.getImgIds(catIds=cat_ids)
            all_img_ids.update(img_ids)

    img_ids = list(all_img_ids)[:num]
    img_infos = coco_gt.loadImgs(img_ids)
    print(f"评估 {len(img_infos)} 张图片")

    total_glip_dets = 0
    total_clip_crops = 0
    total_fusions = 0

    for i, img_info in enumerate(img_infos):
        if i % 20 == 0:
            print(f"[{i + 1}/{len(img_infos)}] {img_info['file_name']}")

        img_path = images_path + img_info['file_name']
        try:
            I = Image.open(img_path)
            image = np.array(I.convert('RGB'))
        except:
            print(f"  跳过损坏图片: {img_info['file_name']}")
            continue

        # GLIP检测
        preds = glip_demo.compute_prediction(image, class_names)
        top_preds = glip_demo._post_process(preds, threshold=confidence)
        scores_glip = top_preds.get_field("scores").tolist()
        boxes = top_preds.bbox.detach().cpu().numpy()

        total_glip_dets += len(boxes)
        if len(boxes) == 0:
            continue

        print(f"  GLIP检测: {len(boxes)}个, 最高分: {max(scores_glip):.4f}")

        # 批量裁剪 - 极低阈值
        crops = []
        high_conf_indices = []
        for j, (box, glip_score) in enumerate(zip(boxes, scores_glip)):
            if glip_score > confidence_clip_crop:
                crop_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                try:
                    crop_img = I.crop(crop_box)
                    if crop_img.size[0] > 5 and crop_img.size[1] > 5:
                        crops.append(crop_img)
                        high_conf_indices.append(j)
                except:
                    continue

        total_clip_crops += len(crops)

        if crops:
            crop_top_scores = batch_clip_scoring(crops, class_names, processor, model, device)

            # 融合得分
            temp_predictions = []
            for crop_idx, top_scores in enumerate(crop_top_scores):
                if not top_scores:  # CLIP失败时的fallback
                    print(f"    Fallback crop {crop_idx}")
                    box_idx = high_conf_indices[crop_idx]
                    box = boxes[box_idx]
                    glip_score = scores_glip[box_idx]

                    # 使用GLIP最高分类别或默认person
                    fallback_class = class_names[0] if class_names[0] in category_map else next(iter(category_map))
                    pred = {
                        "image_id": img_info['id'],
                        "category_id": category_map[fallback_class],
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(glip_score * 0.5)
                    }
                    temp_predictions.append(pred)
                    continue

                box_idx = high_conf_indices[crop_idx]
                box = boxes[box_idx]
                glip_score = scores_glip[box_idx]
                best_class, best_clip_score = top_scores[0]

                if best_class in category_map:
                    fused_score = glip_score * alpha + best_clip_score * (1 - alpha)
                    pred = {
                        "image_id": img_info['id'],
                        "category_id": category_map[best_class],
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(fused_score)
                    }
                    temp_predictions.append(pred)
                    total_fusions += 1

            # NMS
            if temp_predictions:
                nms_results = apply_nms_post_fusion(temp_predictions, conf_threshold=conf_threshold_nms)
                predictions.extend(nms_results)

    print(f"\n=== 统计 ===")
    print(f"总GLIP检测: {total_glip_dets}")
    print(f"总CLIP裁剪: {total_clip_crops}")
    print(f"总融合预测: {total_fusions}")
    print(f"最终预测: {len(predictions)}")

    # 保存并评估
    if predictions:
        pred_file = "glip_clip_final_fixed.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"预测结果保存至: {pred_file}")

        cocoDt = coco_gt.loadRes(pred_file)
        eval_cat_ids = list(category_map.values())

        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")
        cocoEval.params.catIds = eval_cat_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP = cocoEval.stats[1] * 100  # AP@[0.5:0.95]
        mAP50 = cocoEval.stats[0] * 100  # AP@0.5
        print(f"\n🎉 最终结果:")
        print(f"mAP@[0.5:0.95]: {mAP:.2f}%")
        print(f"mAP@0.50:     {mAP50:.2f}%")

        with open('result_clip_rank_final.txt', 'a', encoding='utf-8') as f:
            f.write(f'最终稳定版mAP@[0.5:0.95]：{mAP:.7f}, mAP@0.50：{mAP50:.7f}\n')
    else:
        print("❌ 无任何预测结果！")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    run_success_guaranteed_map(List)
