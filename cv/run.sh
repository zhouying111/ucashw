#!/bin/bash

echo "running begin..."

# 激活虚拟环境并按照相关依赖库
source .venv/Scripts/activate
pip install -r requirements.txt

# 运行 Python 脚本
python main.py

echo "running finished!"
