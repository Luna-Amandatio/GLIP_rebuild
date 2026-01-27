# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys
import urllib.request
import hashlib
from urllib.parse import urlparse
from maskrcnn_benchmark.utils.comm import is_main_process, synchronize


def _download_url_to_file(url, file_name, hash_prefix=None, progress=True):
    """新版PyTorch兼容下载函数"""
    urllib.request.urlretrieve(url, file_name)


def cache_url(url, model_dir='model', progress=True):
    """Loads the Torch serialized object at the given URL."""
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        filename = parts.path.replace("/", "_")

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_url_to_file(url, cached_file, progress=progress)

    synchronize()
    return cached_file
