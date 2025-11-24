# YOLO11-AnchorPedestrianTrack

**基于改进 Anchor 的 YOLO11 行人检测与 ByteTrack 多目标跟踪系统**

------

## 📘 项目简介（Introduction）

本项目实现了一个基于 **改进 YOLO11 模型（Anchor 重设计）** 与 **ByteTrack 多目标跟踪算法** 的行人检测与跟踪系统。
 系统主要用于 **视频监控、自动驾驶、安防场景中的行人识别与ID连续跟踪**。

本项目源于本科毕业设计，包含从 **模型训练 → 检测 → 跟踪 → 可视化展示** 的完整工程流程，结构清晰、模块化设计，易于复现与二次开发。

------

## 🚀 核心功能（Features）

- 🔧 **改进 YOLO11 模型结构**
  - 使用 K-Means 自动生成最适配行人的 Anchor
  - 提升小体积行人、遮挡行人的检测效果
- 🎯 **ByteTrack 多目标跟踪集成**
  - 利用高/低置信度双阶段关联
  - 实现 ID 稳定、连续、不易丢失
- 🧪 **训练、测试、跟踪脚本完整提供**
  - `train.py`：模型训练
  - `detect.py`：单图/视频目标检测
  - `track.py`：行人检测 + 多目标跟踪（ID 输出）
- 📦 **模块化设计（适合工程化与论文复现）**
  - `src/`：数据加载、跟踪工具
  - `models/`：YOLO11 Anchor 改进模型
  - `utils/`：可视化、Anchor 生成器
  - 清晰、干净的项目结构，适合 GitHub 展示与毕业答辩
- 🛡️ **不包含数据集（避免版权问题）**
   请自行准备 VOC/YOLO 格式行人数据集。

------

## 📂 项目结构（Project Structure）

```
YOLO11-AnchorPedestrianTrack/
│── src/               # 数据加载、工具函数、跟踪相关模块
│   ├── dataloader.py
│   ├── model_utils.py
│   └── tracker_utils.py
│
│── models/            # YOLO11 Anchor 改进模型结构
│   └── yolov11_anchor.py
│
│── utils/             # Anchor 生成工具、可视化工具、公共代码
│   ├── anchor_generator.py
│   ├── visualizer.py
│   └── common.py
│
│── data/              # 数据集目录（为空，带 .gitkeep 占位）
│   └── .gitkeep
│
│── train.py           # 模型训练脚本
│── detect.py          # 检测脚本
│── track.py           # 行人检测 + 跟踪脚本
│── requirements.txt   # Python 依赖
│── README.md
│── .gitignore
```

------

## 📦 环境准备（Environment Setup）

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- PyTorch
- Ultralytics（YOLO11）
- OpenCV
- NumPy
- ByteTrack 相关包

------

## 📁 数据集（Dataset）

由于版权限制，本项目不提供数据集。
 请自行准备 **行人检测数据集（VOC 或 YOLO 格式）**，例如：

- COCO 行人类别（person）
- 自动驾驶/监控场景行人数据集
- 商业购买的 VOC 行人数据集

目录结构示例：

```
data/
│── images/
│── labels/
│── train.txt
│── val.txt
```

如果你需要，我可以帮你自动生成 **train/val 划分脚本** 或 **VOC → YOLO 格式转换脚本**。

------

## 🧪 模型训练（Training）

在训练前，需要准备 `your_data.yaml`：

```yaml
path: ./data
train: prepare.txt
val: val.txt
names:
  0: person
```

然后运行：

```bash
python prepare.py
```

默认参数可以直接使用，你也可以根据 GPU 调整：

- `epochs`
- `imgsz`
- `batch`
- `optimizer`
- `workers`

我可以为你写一份优化过的 **4090 训练配置**。

------

## 🔍 行人检测（Detection）

对单张图片：

```bash
python detect.py --source test.jpg
```

对视频：

```bash
python detect.py --source video.mp4
```

检测结果会显示边框与置信度。

------

## 🎥 行人跟踪（Tracking）

运行：

```bash
python track.py --source video.mp4
```

功能：

- 行人 ID 连续跟踪
- 每帧绘制 ID + Bounding Box
- 适合用于论文展示（可以加 IDF1 等指标）

我可以帮你添加 **FPS 显示、轨迹轨迹线条、ID 颜色随机** 等增强效果。

------

## 📊 系统流程图（可用于论文）

如果你需要，我可以帮你画以下图：

- 系统总体架构图
- YOLO11 Anchor 改进模块图
- ByteTrack 跟踪流程图
- 系统部署流程图

这些对毕业答辩非常加分。

------

## 👤 作者信息（Author）

本项目由 **Zeng Jiajun（曾嘉俊，来自广州商学院的一个智能科学与技术专业的学生）** 独立开发，作为本科毕业设计使用。
 代码结构清晰可复现，仅用于学习、研究与展示。

------

## 📜 开源协议（License）

本项目基于 **MIT License** 开源。
 详细内容请查看项目根目录的 `LICENSE` 文件。

MIT 许可允许：

- 自由使用、复制、修改、分发本项目
- 商业用途
- 保留版权声明即可

