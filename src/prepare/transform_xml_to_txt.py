import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

classes = []

def convert(size, box):
    # YOLO 格式转换：归一化后的 x_center, y_center, w, h
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)


def convert_annotation(xml_file, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_filename = root.find("filename").text

    # 自动识别图片格式并读取
    for ext in ["jpg", "png", "jpeg"]:
        img_path = os.path.join(img_dir, img_filename.replace(".jpg", f".{ext}"))
        if os.path.exists(img_path):
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            break
    else:
        print(f"⚠ 找不到图片：{img_filename}")
        return False

    h, w = img.shape[:2]
    results = []

    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in classes:
            classes.append(cls)

        cls_id = classes.index(cls)

        xml_box = obj.find("bndbox")
        xmin = float(xml_box.find("xmin").text)
        xmax = float(xml_box.find("xmax").text)
        ymin = float(xml_box.find("ymin").text)
        ymax = float(xml_box.find("ymax").text)

        bb = convert((w, h), (xmin, xmax, ymin, ymax))
        results.append(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]))

    if len(results) > 0:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        return True
    else:
        return False


if __name__ == "__main__":
    img_dir = "../../data/images"       # 图片的路径
    xml_dir = "../../data/Annotations"  # voc格式标注文件的路径
    out_dir = "../../data/labels"       # 最终生成的txt文件路径

    os.makedirs(out_dir, exist_ok=True)

    error_files = []

    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            xml_file = os.path.join(xml_dir, filename)
            txt_file = os.path.join(out_dir, filename.replace(".xml", ".txt"))

            ok = convert_annotation(xml_file, txt_file)

            if ok:
                print(f"file {filename} convert success.")
            else:
                print(f"file {filename} convert failure.")
                error_files.append(filename)

    # 写入 classes.txt
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    print("\nDataset Classes:", classes)
    print("Error Files:", error_files)
