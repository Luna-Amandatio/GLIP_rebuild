'''
比较模型在各置信度下mAP变化
'''
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到上一级目录（scripts目录）
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
print(f"切换到目录: {os.getcwd()}")
# 将当前目录（scripts）添加到系统路径
sys.path.insert(0, os.getcwd())

import numpy as np
from scripts.GLIP_prediction import GLIPInference



if __name__ == '__main__':
    for confidence in np.arange(0, 1.0, 0.05):
        GLIP = GLIPInference(caption='bird',
                             confidence=confidence,
                             input_num=100,
                             output_file="GLIP模型在各置信度下mAP变化",
                             enable_timer=True)
        GLIP.model_inference()
