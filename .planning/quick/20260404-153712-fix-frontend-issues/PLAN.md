# 执行计划

## 任务概述
修复前端两个交互问题：摄像头停止后仍运行，文件上传后重复弹窗。

## 具体步骤

### 步骤1：分析摄像头停止问题的根本原因
**文件**: `src/run/app.py`
**函数**: `_stream_camera(camera_id: int)`

**当前逻辑**：
1. `cap = cv2.VideoCapture(camera_id)` 打开摄像头
2. `while True:` 无限循环读取帧
3. 通过生成器yield帧数据
4. `finally`块中释放摄像头

**问题**：没有停止机制，即使前端停止请求，循环仍继续。

### 步骤2：实现摄像头停止机制
**方案A**：添加超时机制（较简单）
- 在`while`循环中添加最近帧时间检查
- 如果超过一定时间（如3秒）没有新请求，自动停止
- 缺点：不够精确

**方案B**：使用Flask请求上下文（推荐）
- 检查请求是否仍在连接状态
- 使用`request.environ.get('werkzeug.socket')`或类似方法
- 在读取帧前检查连接是否断开

**方案C**：全局状态管理（复杂但精确）
- 为每个摄像头ID维护运行状态
- 添加停止API端点
- 需要更多架构更改

**选择方案B**：检查请求连接状态，实现简单且有效。

### 步骤3：修改`_stream_camera`函数
**修改位置**: `src/run/app.py`第105-122行

**修改要点**：
1. 获取请求socket对象
2. 在`while`循环中添加连接状态检查
3. 如果连接断开，跳出循环

**伪代码**：
```python
def _stream_camera(camera_id: int):
    # ... 现有代码 ...
    
    try:
        while True:
            # 检查连接是否仍然活动
            # 如果连接断开，跳出循环
            
            success, frame = cap.read()
            if not success:
                break
                
            # ... 处理帧 ...
    finally:
        cap.release()
```

### 步骤4：分析文件上传重复弹窗问题
**相关文件**：
1. `frontend/src/components/panels/ImagePanel.vue`
2. `frontend/src/components/panels/VideoPanel.vue`  
3. `frontend/src/components/ui/FileInput.vue`

**当前流程**：
1. 用户选择文件 → `handleFileChange` → `selectedFile.value = file`
2. 点击检测按钮 → `handleDetect` → `emit('detect', file)` → `fileInputRef.value?.reset()`
3. `reset()`只清空`input.value`，不重置`selectedFile.value`

### 步骤5：修复文件输入重置逻辑
**修改方案**：
1. **ImagePanel.vue/VideoPanel.vue**: 在`handleDetect`中重置`selectedFile.value = null`
2. **FileInput.vue**: 改进`reset()`方法，可能需要触发change事件

**修改要点**：
1. `ImagePanel.vue`: 第39-42行，在`emit('detect', file)`后重置`selectedFile.value = null`
2. `VideoPanel.vue`: 第41-47行，同样重置`selectedFile.value = null`
3. 可选：`FileInput.vue`的`reset()`方法也可以触发一个空change事件

### 步骤6：验证修改
1. **摄像头测试**：
   - 启动摄像头，确认流正常
   - 停止摄像头，确认后端循环停止
   - 检查Flask日志输出

2. **文件上传测试**：
   - 选择文件并开始检测
   - 确认文件选择对话框不自动弹出
   - 可以正常选择新文件

## 风险与缓解
- **风险**: 连接状态检查可能影响性能
- **缓解**: 每N帧检查一次，不是每帧
- **风险**: 文件重置可能影响用户体验
- **缓解**: 保持当前选择直到新检测开始

## 成功标准
1. `_stream_camera`函数添加连接状态检查
2. 前端面板正确重置文件选择状态
3. 摄像头停止后后端循环终止
4. 文件上传后不再自动弹窗