# 快速任务：修复视频检测TypeError错误

## 问题描述
视频检测时出现TypeError: 'NoneType' object is not subscriptable错误。经过分析，问题可能由`process_video_file`方法中的`persist=True`参数引起。

## 根本原因
在`src/run/tracker.py`文件的`process_video_file`方法中，第153行添加了`persist=True`参数。这个参数可能导致YOLO模型返回`None`，从而引发TypeError。

## 解决方案
移除`persist=True`参数，恢复为main分支的配置。

## 具体修改
文件：`src/run/tracker.py`
方法：`process_video_file`
修改：删除第153行的`persist=True,`参数

## 验证方法
1. 运行视频检测功能，确认不再出现TypeError
2. 确保所有现有功能正常工作

## 影响范围
- 仅影响视频文件处理功能
- 不影响图像检测和实时摄像头检测
- 与现有Flask API完全兼容