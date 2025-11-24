import os
import shutil
from sklearn.model_selection import train_test_split

# 参数设置
val_size = 0.1
postfix = 'jpg'
imgpath = r'E:\all_project\pythonProject\yolo11\data_mask\images'
txtpath = r'E:\all_project\pythonProject\yolo11\label\labels'
output_train_img_folder = r'E:\all_project\pythonProject\yolo11\ultralytics-main\dataset\train\images'
output_val_img_folder = r'E:\all_project\pythonProject\yolo11\ultralytics-main\dataset\valid\images'
output_train_txt_folder = r'E:\all_project\pythonProject\yolo11\ultralytics-main\dataset\train\labels'
output_val_txt_folder = r'E:\all_project\pythonProject\yolo11\ultralytics-main\dataset\valid\labels'

# 创建目标文件夹
os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)

# 检查文件匹配
image_files = set(f.split('.')[0] for f in os.listdir(imgpath) if f.endswith(postfix))
label_files = set(f.split('.')[0] for f in os.listdir(txtpath) if f.endswith('txt'))

unmatched_labels = label_files - image_files
unmatched_images = image_files - label_files

# 打印警告信息，但程序继续执行
if unmatched_labels:
    print(f"以下标签文件没有匹配的图像，将跳过：{unmatched_labels}")
if unmatched_images:
    print(f"以下图像文件没有匹配的标签，将跳过：{unmatched_images}")

# 只保留匹配的文件
matched_files = list(label_files & image_files)

# 数据集划分
train, val = train_test_split(matched_files, test_size=val_size, shuffle=True, random_state=0)

# 文件复制函数
def copy_files(file_list, src_img_path, src_txt_path, dest_img_folder, dest_txt_folder):
    for file_name in file_list:
        img_src = os.path.join(src_img_path, f"{file_name}.{postfix}")
        txt_src = os.path.join(src_txt_path, f"{file_name}.txt")
        img_dest = os.path.join(dest_img_folder, f"{file_name}.{postfix}")
        txt_dest = os.path.join(dest_txt_folder, f"{file_name}.txt")

        if os.path.exists(img_src) and os.path.exists(txt_src):
            shutil.copy(img_src, img_dest)
            shutil.copy(txt_src, txt_dest)
        else:
            print(f"文件 {file_name} 的图像或标签缺失，跳过...")

# 执行文件复制
copy_files(train, imgpath, txtpath, output_train_img_folder, output_train_txt_folder)
copy_files(val, imgpath, txtpath, output_val_img_folder, output_val_txt_folder)

print("数据集划分完成！")

