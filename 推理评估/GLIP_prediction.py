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
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from tqdm import tqdm

from dataset_information import dataset_info

class GLIPInference:
    """

    glip推理封装,支持单类别与多类别输入

    Attributes:
        caption : 输入glip模型的提示词
        confidence : glip的置信度阈值
        num : 推理的图片数量
        #batch : 同时推理批次
        model : glip模型

        dataset : 数据集名
        mode : 模式，决定选择数据集中的什么目录(train or test or valid)
        images_dir : 图片所处目录，由dataset与mode决定

        json_path : 数据集配置的json文件位置
        category_map : 类别映射到ID字典

    Methods:
        model_load  : 负责glip模型加载与编译
        dataset_load : 负责数据集图片加载与映射
        model_inference : 负责glip模型推理
        mAP_Calculation : 对生成的.json文件评估mAP等指标
        warmup_model : 模型预热

    """
    def __init__(self,
                 caption,
                 confidence = 0.5,
                 dataset = 'Coco.v1i.coco-segmentation_2',
                 input_num = 100,
                 mode = "valid",
                 output_file = "mAP_results.json",):

        print(f"选择的图片目录为：{dataset}/{mode}")
        print(f"输入的置信度为{confidence}")

        self.confidence = confidence
        self.num = input_num
        if isinstance(caption, str):
            self.caption = [caption]
        else:
            self.caption = caption
        self.model = self.model_load()

        self.dataset = dataset
        self.mode = mode
        self.images_dir = "../DATASET/" + dataset +"/" + mode + "/"

        self.json_path = None
        self.output_file = output_file
        self.category_map = {}

        self.img_infos  = self.dataset_load()


    def model_load(self):
        """加载GLIP模型"""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # GLIP配置
        cfg_file = r"..\configs\pretrain\glip_Swin_T_O365_GoldG.yaml"
        weight_file = r"..\MODEL\glip_tiny_model_o365_goldg_cc_sbu.pth"

        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(cfg_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", device])

        glip_demo = GLIPDemo(cfg, min_image_size=800, confidence_threshold=self.confidence)
        glip_demo.model.eval()

        #pytorch2.0编译加速
        torch.compile(model=glip_demo.model, mode="max-autotune",backend="cudagraphs")
        self.warmup_model(glip_demo)

        return glip_demo


    def dataset_load(self):
        """加载数据集"""
        img_dir = self.dataset + "/" + self.mode
        try:
            self.json_path = dataset_info.json_files[img_dir]
        except Exception as e:
            print("输入路径：\n",img_dir)
            print(f"生成的数据集名与数据集路径字典：{dataset_info.json_files}\n")
            print(f"请仔细校验{e}")
            raise

        coco_gt = COCO(self.json_path)

        # 类别映射到ID
        for class_name in self.caption:
            cat_ids = coco_gt.getCatIds(catNms=[class_name])
            self.category_map[class_name] = cat_ids[0]

        # 获取图像ID，用集合去重
        all_img_ids = set()
        for class_name in self.caption:
            cat_ids = coco_gt.getCatIds(catNms=[class_name])

            img_ids = coco_gt.getImgIds(catIds=cat_ids)
            all_img_ids.update(img_ids)

        img_ids = list(all_img_ids)[:self.num]
        img_infos = coco_gt.loadImgs(img_ids)

        return img_infos


    def model_inference(self):
        """模型推理"""
        predictions = []
        length = len(self.img_infos)
        pbar = tqdm(self.img_infos,
                    desc="推理进度",
                    unit="it",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # 创建从label_id到category_id的映射
        # GLIP的label_id从1开始，对应self.caption列表中的索引
        label_to_catid = {}
        for i, class_name in enumerate(self.caption):
            # label_id = i + 1 对应 self.caption[i]
            if class_name in self.category_map:
                label_to_catid[i + 1] = self.category_map[class_name]

        print(f"标签映射关系: {label_to_catid}")
        print(f"类别名称到ID映射: {self.category_map}")

        for i, img_info in enumerate(pbar):
            try:
                img_path = self.images_dir + img_info['file_name']
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            except:
                continue

            print(f"[{i + 1}/{length}] {img_info['file_name']})")

            # 半精度推理
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds = self.model.compute_prediction(image, self.caption)
                top_preds = self.model._post_process(preds, threshold=self.confidence)

            labels = top_preds.get_field("labels").tolist()
            scores = top_preds.get_field("scores").tolist()
            boxes = top_preds.bbox.detach().cpu().numpy()

            # 保存预测结果
            for box, score, label_id in zip(boxes, scores, labels):
                pred = {
                    "image_id": img_info['id'],
                    "category_id": label_to_catid[label_id],
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "score": float(score)
                }
                predictions.append(pred)

            if len(labels) > 0:
                print(f"检测到 {len(labels)} 个目标")

            else:
                print(f"未检测到目标")

        # 保存预测结果
        pred_file = "预测结果.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)


    def mAP_Calculation(self):
        """mAP计算"""
        coco_gt = COCO(self.json_path)
        cocoDt = coco_gt.loadRes("预测结果.json")

        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")

        eval_cat_ids = list(self.category_map.values())
        cocoEval.params.catIds = eval_cat_ids
        cocoEval.params.imgIds = [img_info["id"] for img_info in self.img_infos]  # 只评估有标注的图片

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results = {
            "mAP": float(cocoEval.stats[0]),
            "AP_50": float(cocoEval.stats[1]),
            "AP_75": float(cocoEval.stats[2]),
            "AP_small": float(cocoEval.stats[3]),
            "AP_medium": float(cocoEval.stats[4]),
            "AP_large": float(cocoEval.stats[5]),
            "AR_1": float(cocoEval.stats[6]),
            "AR_10": float(cocoEval.stats[7]),
            "AR_100": float(cocoEval.stats[8]),
            "AR_small": float(cocoEval.stats[9]),
            "AR_medium": float(cocoEval.stats[10]),
            "AR_large": float(cocoEval.stats[11]),
            "categories": {
                name: cat_id for name, cat_id in self.category_map.items()
            },
            "num_images": len(self.img_infos),
            "confidence_threshold": self.confidence
        }
        # 保存JSON
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"结果已保存到 {self.output_file}")

        return 1


    def warmup_model(self, model, num_iterations=5):
        """预热GLIP模型"""
        print("开始预热模型...")

        # 创建预热用的虚拟图像
        dummy_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

        # 使用预热caption
        warmup_caption = self.caption if self.caption else ["person"]

        print(f"预热迭代次数: {num_iterations}")

        # 预热循环
        for i in range(num_iterations):
            print(f"预热进度: {i + 1}/{num_iterations}")

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    try:
                        preds = model.compute_prediction(dummy_image, warmup_caption)
                        _ = model._post_process(preds, threshold=self.confidence)
                    except Exception as e:
                        print(f"预热过程中出现错误: {e}")
                        continue

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("模型预热完成！")
def clean():
    for file in Path('.').glob('*.json'):
        file.unlink()
        print(f"已删除: {file}")

#-------------------------
#测试
#-------------------------
if __name__ == "__main__":

    glip = GLIPInference(["bird","person"],0.3)
    glip.model_inference()
    glip.mAP_Calculation()

    #clean()