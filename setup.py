
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

from wheel.bdist_wheel import bdist_wheel

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "nni"

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))


setup(
    name=PACKAGE_NAME,
    version="1.0",
    packages=find_packages(
        exclude=(
            "data",
            "nas",
            "nn_meter",
            "nni_node",
            "training_hf",
        )
    ),
    author="Hoang-Loc La",
    author_email="hoang.l.la@uit.no",
    description="EnergyNAS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enyac-group/Quamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "colorama",
        "typeguard==2.13",
        "scikit-learn",  ## 1.0.1
        "filelock==3.10",
        "cloudpickle",
        "json_tricks",
        "PyYAML",
        "psutil",
        "requests",
        "astor",
        "schema",
        "pytorch_lightning",
        "websockets",
        "onnxsim",
        "onnx",
        "jsonlines",
        "json_tricks",
        "onnxruntime",
        "numpy"
    ],
)
