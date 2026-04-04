# 执行计划

## 任务概述
修复视频检测中的TypeError错误，通过移除`persist=True`参数恢复为main分支配置。

## 具体步骤

### 步骤1：定位并修改代码
- 文件：`src/run/tracker.py`
- 方法：`process_video_file`（第119行开始）
- 修改位置：第153行
- 操作：删除`persist=True,`这一行

**修改前：**
```python
            results = self.model.track(
                source=str(input_path),
                persist=True,
                tracker=self.tracker_config,
```

**修改后：**
```python
            results = self.model.track(
                source=str(input_path),
                tracker=self.tracker_config,
```

### 步骤2：验证修改
1. 检查语法是否正确
2. 确保缩进一致
3. 确认没有破坏其他参数

### 步骤3：创建原子提交
- 提交信息：`fix(video): remove persist=True parameter to fix TypeError`
- 只提交`src/run/tracker.py`文件的修改

### 步骤4：更新状态
更新`.planning/STATE.md`中的"Quick Tasks Completed"表格。

## 风险与缓解
- **风险**：移除参数可能影响跟踪连续性
- **缓解**：`persist`参数默认为`False`，符合main分支原有行为
- **验证**：测试视频检测功能是否正常工作

## 成功标准
- 代码修改正确完成
- 提交创建成功
- STATE.md更新完成