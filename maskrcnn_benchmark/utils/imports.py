import torch
import sys
import importlib.util


def import_file(module_name, file_path, make_importable=False):
    """Import a python file given its path. 兼容所有Python版本"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    if make_importable:
        sys.modules[module_name] = module

    spec.loader.exec_module(module)
    return module
