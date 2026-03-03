

### 项目结果

![本地图片](.\assert\GLIP使用模版句与类别提示mAP差异.png "本地图片示例")



---



### 项目结构

├─assert          ----示例图片资源    
├─DATASET    ----数据集存放目录    
├─logs             ----profile生成的结果    
├─models       ----模型存放位置    
├─scripts         ----辅助脚本   
└─结果保存     ----可视化生成图片   



---



### 依赖安装

```bash
./编译安装.bat
```

GLIP模型权重地址 ： https://huggingface.co/GLIPModel/GLIP/blob/main/glip_tiny_model_o365_goldg_cc_sbu.pth
COCO数据集下载位置 ：https://app.roboflow.com/ds/qHK8Q42lc8?key=27IHm68o5o

---



### scripts 内各文件功能

1. CLIPRerank :内包含CLIP与GroundDINO混合模型推理类
2. dataset_information ： 内包含数据集信息查询类
3. GroundDINO_prediction ：内包含GroundDINO单模型推理类
4. KeywordGenerate ： 基于blip与nltk分词的关键词生成,用于图片关键词自动标注



---



### CLIP重排功能解析



改进动机：针对GLIP模型在实际场景中存在的类别误识别问题，提出基于CLIP的类别重打分机制。GLIP虽然能准确定位目标，但在复杂场景下类别预测准确率有限；CLIP基于大规模图文预训练，具备更强的语义理解和泛化能力。本方法融合两者优势：保留GLIP的定位能力，同时利用CLIP对每个候选框进行类别重判，最终通过加权融合（alpha参数）得到更准确的分类结果。

 

推理流程：

1. GLIP检测：输入图像，生成候选框及初始置信度

2. 图像裁剪：根据检测框裁剪目标区域

3. CLIP分类：对裁剪区域进行图文匹配，计算各类别概率

4. 分数融合：加权融合GLIP分数和CLIP分数（alpha参数）

   

#### 效果

**未进行CLIP重排结果**

![本地图片](.\assert\未进行CLIP重排结果.jpg "本地图片示例")

**进行CLIP重排结果**

![本地图片](.\assert\进行CLIP重排结果.jpg "本地图片示例")



---



### NMS去重

方法说明：本方法对GLIP模型输出的初始检测结果进行精细化后处理。通过调用PyTorch框架中的nms函数，基于检测框的IoU（交并比）阈值对高度重叠的候选框进行筛选去重，仅保留置信度最高的检测框。这一步骤能够有效减少冗余检测，提升最终预测结果的准确性和可读性。

 

核心流程：

1. 收集GLIP模型在各个图片上检测到的目标框及其置信度分数
2. 调用torchvision.ops.nms函数，根据预设的IoU阈值（self.iou_threshold）对同一类别内的检测框进行非极大值抑制
3. 结合置信度阈值（conf_threshold）过滤低质量检测框
4. 返回经过筛选的精简预测结果，用于后续的mAP评估

 

*效果：*

**未进行NMS去重**

![本地图片](.\assert\nms未开启结果1.jpg "本地图片示例")

进行NMS去重

![本地图片](.\assert\nms开启后结果1.jpg "本地图片示例")



---



### 其余创新

1. 对GroundDINO模型进行了pytorch2.0编译，使模型提速

2. 将GroundDINO模型推理使用半精度推理优化


   

