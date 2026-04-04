# 执行计划

## 任务概述
修复Flask CORS配置和视频编码器兼容性问题，解决前端Network Error和编码器警告。

## 具体步骤

### 步骤1：添加flask-cors依赖
**文件**: `requirements.txt`
**操作**: 在Flask依赖后添加`flask-cors`包

**修改前**:
```txt
flask>=3.0.0

# Data preparation utilities
scikit-learn>=1.3.0
```

**修改后**:
```txt
flask>=3.0.0
flask-cors>=4.0.0

# Data preparation utilities
scikit-learn>=1.3.0
```

### 步骤2：安装依赖
**命令**: `pip install flask-cors`
**说明**: 如果已安装则跳过

### 步骤3：修改Flask应用添加CORS支持
**文件**: `src/run/app.py`
**修改**: 导入flask_cors并初始化CORS中间件

**修改位置**: 在Flask导入后，app创建后

**修改前**:
```python
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# ... 其他导入 ...

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
```

**修改后**:
```python
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ... 其他导入 ...

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
CORS(app)  # 启用CORS支持，允许所有跨域请求
```

### 步骤4：修复视频编码器
**文件**: `src/run/tracker.py`
**方法**: `process_video_file`
**修改**: 将编码器从`MJPG`改为`mp4v`

**修改位置**: 第140行左右

**修改前**:
```python
        # 创建 VideoWriter - 使用 MJPG 编码（稳定兼容）
        output_path_str = str(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
```

**修改后**:
```python
        # 创建 VideoWriter - 使用 mp4v 编码（MP4兼容）
        output_path_str = str(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
```

### 步骤5：验证修改
1. 检查语法是否正确
2. 确保所有修改文件缩进一致
3. 测试Flask应用启动无错误
4. 验证前端CORS错误消失

## 风险与缓解
- **风险**: CORS(app)允许所有来源，生产环境需限制
- **缓解**: 当前为开发环境，可接受宽松配置
- **风险**: 编码器更改可能影响视频质量
- **缓解**: mp4v是标准MP4编码器，广泛兼容

## 成功标准
1. `flask-cors`添加到requirements.txt
2. `app.py`正确配置CORS
3. `tracker.py`编码器改为mp4v
4. 前端不再出现CORS错误
5. 视频检测无编码器警告