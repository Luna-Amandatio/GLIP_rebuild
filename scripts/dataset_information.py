"""
获取数据集信息
"""
import json
import os
from pprint import pprint

def get_subdirs_scandir(path):
    """使用os.scandir获取子目录名"""
    with os.scandir(path) as entries:
        subdirs = [entry.name for entry in entries if entry.is_dir()]

    if len(subdirs) == 0:
        print("未找到任何目录")
        exit()

    return subdirs

class DatasetInfo:
    """
    在项目目录DATASET中获取数据集信息
    Attributes:
        base_path:数据集基础目录
        dir_name :获取到的数据集名
        json_files:获取到的json文件路径字典

    Methods:
        get_subdirs_scandi:获取子目录名
        json_finder:搜索数据集中json文件
        ImformationPrint:打印json文件信息
    """
    def __init__(self):
        print("正在进行数据集加载与校验")
        self.base_path = "../DATASET"
        self.dir_name = get_subdirs_scandir(self.base_path)
        self.json_files = self.json_finder()


    def json_finder(self):
        json_files = {}
        for dir in self.dir_name:
            mode_path = self.base_path + "/" + dir
            sub_dirs = get_subdirs_scandir(mode_path)

            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(mode_path, sub_dir)
                #查找目录下所有JSON文件

                for file in os.listdir(sub_dir_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(sub_dir_path, file)
                        json_files[f"{dir}/{sub_dir}"] = file_path
                        #print(f"找到JSON文件: {file_path}")
        return json_files

    def ImformationPrint(self):
        for key, value in self.json_files.items():
            json_path = value

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查你的数据集中哪些类别存在
                your_categories = {cat['name']: cat['id'] for cat in data['categories']}
                pprint(f"{key}数据集中类别: {list(your_categories.keys())}")


            except FileNotFoundError:
                print(f"  警告: 未找到标注文件 - {json_path}")
                continue

            except json.JSONDecodeError:
                print(f"  警告: JSON文件格式错误 - {json_path}")
                continue

            except KeyError as e:
                print(f"  警告: JSON文件缺少关键字段 {e} - {json_path}")
                continue

dataset_info = DatasetInfo()
#---------------------------------
#测试
# ---------------------------------
if __name__ == "__main__":
    #dataset_info.ImformationPrint()
    print(dataset_info.json_files)