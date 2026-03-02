# YOLO11-AnchorPedestrianTrack

基于 `YOLO11 + CBAM + ByteTrack + Flask` 的行人检测与多目标跟踪项目，包含数据预处理、训练、实时跟踪和网页可视化。

## 1. 项目目标

本项目用于行人场景下的目标检测与跟踪，支持：

- 基于 YOLO11 的行人检测（类别 `person`）
- 基于 ByteTrack 的多目标跟踪（ID 连续）
- 在视频流中叠加 `Person Count` 和 `FPS`
- 通过 Flask 网页实时展示结果

## 2. 当前代码结构（按实际仓库）

```text
YOLO11-AnchorPedestrianTrack/
├─ src/
│  ├─ prepare/
│  │  ├─ transform_xml_to_txt.py      # VOC XML -> YOLO TXT
│  │  ├─ take_train_and_valid.py      # 数据集划分（train/valid/test）
│  │  ├─ cuda_test.py                 # CUDA 可用性测试
│  │  └─ test_opencv.py               # OpenCV/摄像头测试（示例脚本）
│  ├─ run/
│  │  ├─ train.py                     # 训练入口（YOLO11 + CBAM 混合方案）
│  │  ├─ tracker.py                   # 跟踪逻辑（ByteTrack + FPS/人数叠加）
│  │  └─ app.py                       # Flask Web 服务入口
│  └─ utils/
│     ├─ cbam.py
│     └─ losses.py
├─ dataset/
│  ├─ train/images, train/labels
│  ├─ valid/images, valid/labels
│  ├─ test/images,  test/labels
│  └─ dataset.yaml
├─ frontend/
│  └─ index.html
├─ models/                            # 模型配置/权重目录
├─ result/                            # 训练产出目录
├─ ultralytics/                       # 本地 ultralytics 源码（可选）
└─ requirements.txt
```

## 3. 环境要求

- Windows + Anaconda（推荐）
- Python 3.10/3.11
- NVIDIA GPU（可选，但训练强烈推荐）
- CUDA 12.1（与当前 `requirements.txt` 中 torch 版本对应）

## 4. 依赖安装

在项目根目录执行：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含：

- `torch/torchvision/torchaudio`（CUDA 12.1）
- `ultralytics`
- `opencv-python`
- `numpy`
- `PyYAML`
- `flask`
- `scikit-learn`

## 5. 数据准备流程

### 5.1 标注转换（VOC XML -> YOLO TXT）

脚本：`src/prepare/transform_xml_to_txt.py`

默认目录约定：

- 输入图片：`data/images`
- 输入标注：`data/Annotations`
- 输出标签：`data/labels`

运行：

```bash
python src/prepare/transform_xml_to_txt.py
```

### 5.2 数据集划分（train/valid/test）

脚本：`src/prepare/take_train_and_valid.py`

默认比例：

- train: 70%
- valid: 20%
- test: 10%

运行：

```bash
python src/prepare/take_train_and_valid.py
```

输出到 `dataset/` 目录下对应子目录。

### 5.3 数据配置文件

文件：`dataset/dataset.yaml`

当前示例配置：

```yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['person']
```

## 6. 训练

训练入口：

```bash
python src/run/train.py
```

说明：

- 脚本会优先尝试使用仓库内 `ultralytics/`（如果结构完整）
- 若本地源码不完整，则回退到系统安装的 `ultralytics`
- 训练数据路径默认读取 `dataset/dataset.yaml`
- 训练输出在 `result/hybrid_weights/` 下

## 7. 实时跟踪与网页演示

### 7.1 启动方式

```bash
python src/run/app.py
```

访问：

- `http://127.0.0.1:5000`

### 7.2 当前 Web 端行为

- 首页模板：`frontend/index.html`
- 视频流接口：`/video_feed`
- 画面叠加信息由 `src/run/tracker.py` 提供：
  - `Person Count`
  - `FPS`（指数平滑）

### 7.3 常用修改点

文件：`src/run/app.py`

- 模型权重路径：`model_path`
- 视频输入源：`video_source`
  - `0` 表示默认摄像头
  - 也可改成视频文件路径
- 服务端口：`PORT`

## 8. 已处理的关键问题

- `README` 编码问题：已调整为可在 Windows PowerShell 下正常显示
- Flask 启动日志端口与真实端口不一致：已统一
- 模板路径相对定位问题：已改为基于文件位置的绝对路径
- `python src/run/app.py` 导入失败（`No module named run`）：已兼容脚本模式与模块模式
- 视频流画面已支持 FPS 显示

## 9. 常见问题（FAQ）

### Q1: 为什么静止后检测框偶尔消失？

常见原因：

- `conf` 阈值较高导致检测帧被过滤
- ByteTrack 关联阈值与轨迹缓存策略导致短暂丢轨

可调参数位置：`src/run/tracker.py`

- `self.conf_threshold`
- `self.iou_threshold`
- `self.tracker_config`（可切换到自定义 `bytetrack.yaml`）

### Q2: 为什么能打开网页但看不到视频？

排查顺序：

1. 检查摄像头是否被其他软件占用
2. 将 `video_source=0` 改为本地视频路径测试
3. 确认模型权重路径存在且可加载
4. 检查 `opencv-python`、`ultralytics` 是否安装成功

### Q3: `favicon.ico 404` 是错误吗？

不是。浏览器会自动请求网站图标，不影响检测/跟踪主功能。

## 10. 毕设建议（可直接用于答辩材料）

建议最少补齐这三组证据：

1. 定量指标：
   `mAP50/mAP50-95`（检测）+ `IDF1/MOTA`（跟踪）
2. 对比实验：
   基线 YOLO11 vs 加 Anchor/CBAM vs 加 ByteTrack
3. 可视化展示：
   实时网页演示 + 典型成功/失败案例分析

## 11. 许可证

本项目采用 `MIT License`，详见 `LICENSE`。
