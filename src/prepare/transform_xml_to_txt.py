'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-12-04 17:51:57
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-03-27 09:23:15
FilePath: \pythonProject\yolo11\ultralytics-main\scripts\transform_xml_to_txt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 这段代码用于将之前的xml文件转换为txt文件，用来训练yolo模型
import cv2
import os
import xml.etree.ElementTree as ET

import numpy as np

classes = []


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))


if __name__ == "__main__":
    postfix = 'jpg'  # 图像后缀，根据自己的图像后缀修改
    imgpath = r'E:\all_project\pythonProject\yolo11\data_mask\images'  # 图像文件路径
    xmlpath = r'E:\all_project\pythonProject\yolo11\data_mask\images'  # xml文件文件路径
    txtpath = r'E:\all_project\pythonProject\yolo11\label\labels'  # 生成的txt文件路径

    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)

    list = os.listdir(xmlpath)
    error_file_list = []
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i])
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
        except Exception as e:
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(list[i])
    print(f'this file convert failure\n{error_file_list}')
    print(f'Dataset Classes:{classes}')
