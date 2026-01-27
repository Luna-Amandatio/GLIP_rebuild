::GLIP依赖安装
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers

::CLIP依赖安装
pip install torch torchvision
pip install regex tqdm
pip install git+https://github.com/openai/CLIP.git

::编译
pip install -e . --no-build-isolation

::pip缓存清理
pip cache purge

::构建是否成功检测
python -c "from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo; print('构建成功！')"