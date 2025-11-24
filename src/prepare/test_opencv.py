import cv2
from ultralytics import YOLO
# 这个用于测试电脑上的摄像头能不能用
# 加载 YOLO 模型
model = YOLO(r'E:\all_project\pythonProject\yolo11\ultralytics-main\yolo11n.pt')  # 使用 YOLOv8 的官方预训练模型

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头图像！")
        break

    # YOLO 目标检测
    results = model.predict(source=frame, conf=0.5)
    result_frame = results[0].plot()

    # 显示结果
    cv2.imshow("YOLO 检测", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
        break

cap.release()
cv2.destroyAllWindows()
