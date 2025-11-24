import os
import shutil
from sklearn.model_selection import train_test_split

# ==============================
# 参数设置
# ==============================

postfix = 'jpg'

imgpath = '../../data/images'
txtpath = '../../data/labels'

output_train_img_folder = '../../dataset/train/images'
output_val_img_folder = '../../dataset/valid/images'
output_test_img_folder = '../../dataset/test/images'

output_train_txt_folder = '../../dataset/train/labels'
output_val_txt_folder = '../../dataset/valid/labels'
output_test_txt_folder = '../../dataset/test/labels'

# ==============================
# 创建目标文件夹
# ==============================
os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_test_img_folder, exist_ok=True)

os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)
os.makedirs(output_test_txt_folder, exist_ok=True)

# ==============================
# 检查文件匹配情况
# ==============================
image_files = set(f.split('.')[0] for f in os.listdir(imgpath) if f.endswith(postfix))
label_files = set(f.split('.')[0] for f in os.listdir(txtpath) if f.endswith('txt'))

unmatched_labels = label_files - image_files
unmatched_images = image_files - label_files

if unmatched_labels:
    print(f"以下标签文件没有匹配的图像，将跳过：{unmatched_labels}")
if unmatched_images:
    print(f"以下图像文件没有匹配的标签，将跳过：{unmatched_images}")

# 得到匹配的文件名列表
matched_files = list(label_files & image_files)

# ==============================
# 三方划分：train 70%, val 20%, test 10%
# ==============================

# 先划分出 train（70%） + temp（30%）
train_files, temp_files = train_test_split(
    matched_files,
    test_size=0.30,
    shuffle=True,
    random_state=0
)

# 再把 temp 划分为 val（20%）和 test（10%）
# temp 占 30%，其中 test 需要占 10%，即：10/30 = 0.3333
val_files, test_files = train_test_split(
    temp_files,
    test_size=(1/3),
    shuffle=True,
    random_state=0
)

print(f"训练集数量: {len(train_files)}")
print(f"验证集数量: {len(val_files)}")
print(f"测试集数量: {len(test_files)}")

# ==============================
# 文件复制函数
# ==============================
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
            print(f"文件 {file_name} 缺失图像或标签，跳过...")

# ==============================
# 执行文件复制
# ==============================
copy_files(train_files, imgpath, txtpath, output_train_img_folder, output_train_txt_folder)
copy_files(val_files, imgpath, txtpath, output_val_img_folder, output_val_txt_folder)
copy_files(test_files, imgpath, txtpath, output_test_img_folder, output_test_txt_folder)

print("\n数据集划分完成！")
