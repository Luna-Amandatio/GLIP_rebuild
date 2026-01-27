import json

#数据集基础目录
base_path = "D:/Project/ComSen/GLIP/DATASET/Coco.v1i.coco-segmentation_2"
json_path = base_path + "/valid/_annotations.coco.json"

HIGH_SUCCESS_CLASSES = ["person", "car", "dog", "bicycle", "cat"]


with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查你的数据集中哪些类别存在
your_categories = {cat['name']: cat['id'] for cat in data['categories']}
valid_classes = [cls for cls in HIGH_SUCCESS_CLASSES if cls in your_categories]
print(f"你的类别: {list(your_categories.keys())}")
print(f"高成功率类别: {valid_classes}")