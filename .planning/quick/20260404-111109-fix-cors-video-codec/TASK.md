# 快速任务：修复Flask CORS配置和视频编码器

## 问题描述

### 1. CORS错误
前端从`http://127.0.0.1:5000`请求`http://localhost:5000`时被跨域策略阻止：
```
Access to XMLHttpRequest at 'http://localhost:5000/detect/video' from origin 'http://127.0.0.1:5000' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

### 2. 视频编码器警告
视频处理时出现编码器不兼容警告：
```
OpenCV: FFMPEG: tag 0x47504a4d/'MJPG' is not supported with codec id 7 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
```

## 根本原因

### CORS问题
Flask应用未配置CORS头，浏览器拒绝跨域请求。尽管127.0.0.1和localhost指向同一地址，但浏览器视为不同源。

### 视频编码器问题
视频输出使用`.mp4`扩展名，但编码器使用`MJPG`（Motion JPEG），与MP4容器不兼容。OpenCV自动回退到`mp4v`编码器。

## 解决方案

### 1. 添加Flask CORS支持
- 安装`flask-cors`包
- 在`src/run/app.py`中添加CORS配置
- 允许来自所有源的请求（开发环境）

### 2. 修复视频编码器
- 修改`src/run/tracker.py`中的编码器
- 将`MJPG`改为`mp4v`（MPEG-4视频编码）
- 确保编码器与`.mp4`文件格式兼容

## 验证方法
1. 启动Flask应用后，前端不再出现CORS错误
2. 视频检测功能正常工作，无编码器警告
3. 网络请求成功完成，返回检测结果