"""
Vercel Serverless API 入口
兼容 Vercel Serverless 部署
"""

import os
import sys
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 设置环境变量（关键！）
os.environ['VERCEL_ENV'] = 'true'

# 导入主应用
from app import app

# Vercel需要app作为默认导出
# 对于Flask应用，Vercel会自动处理WSGI接口