#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLO11+ByteTrack的行人检测与跟踪系统 - 训练脚本
论文: 基于YOLO11+ByteTrack的行人检测与跟踪系统设计与实现

本脚本使用本地ultralytics源代码，支持CBAM注意力机制集成
适用于学术研究和算法改进
"""

import os
import sys
import yaml
import torch

# 确保使用本地ultralytics源代码（用于研究和改进）
current_dir = os.getcwd()
local_ultralytics = os.path.join(current_dir, 'ultralytics')

# 将本地ultralytics添加到Python路径的最前面
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

print(f"📚 使用本地ultralytics源代码: {local_ultralytics}")

try:
    from ultralytics import YOLO
    print("✅ 成功导入本地ultralytics包")
    
    # 验证是否使用了本地版本
    import ultralytics
    ultralytics_path = ultralytics.__file__
    if local_ultralytics in ultralytics_path:
        print(f"✅ 确认使用本地ultralytics: {ultralytics_path}")
    else:
        print(f"⚠️ 警告: 可能使用了系统ultralytics: {ultralytics_path}")
        
except ImportError as e:
    print(f"❌ 无法导入ultralytics: {e}")
    print("请确保本地ultralytics源代码完整")
    raise

# CBAM utils - 使用本地实现
from src.utils.cbam import CBAMWrapper, CBAM
# optional losses (we will use Focal via hyperparams)
from src.utils.losses import FocalLoss, siou_loss

# ============= 研究配置 =============
DATA_YAML = "./dataset/dataset.yaml"  # 行人检测数据集配置
MODEL_PATH = "./models/YOLO11m.pt"  # 预训练模型
CBAM_MODEL_YAML = "./models/yolo11m_cbam.yaml"  # CBAM增强配置
SAVE_DIR = "result/research_weights"  # 研究结果保存目录
CBAM_INSERT_LAYERS = [5, 8]  # CBAM插入位置（基于网络分析确定）

# 研究模式选择
USE_CONFIG_FILE = True  # True: 配置文件方式, False: 动态插入方式
RESEARCH_MODE = "cbam_integration"  # 研究模式标识

# 实验配置
EXPERIMENT_NAME = "YOLO11m_CBAM_Pedestrian"
PAPER_VERSION = "v1.0"
# =================================

os.makedirs(SAVE_DIR, exist_ok=True)

def log_research_info():
    """记录研究信息用于论文"""
    info = {
        "paper_title": "基于YOLO11+ByteTrack的行人检测与跟踪系统设计与实现",
        "experiment_name": EXPERIMENT_NAME,
        "version": PAPER_VERSION,
        "model_base": "YOLO11m",
        "enhancement": "CBAM Attention Mechanism",
        "dataset": "Pedestrian Detection Dataset",
        "research_mode": RESEARCH_MODE,
        "cbam_layers": CBAM_INSERT_LAYERS,
        "ultralytics_source": "Local (Modified for Research)"
    }
    
    info_path = os.path.join(SAVE_DIR, "research_info.yaml")
    with open(info_path, "w", encoding='utf-8') as f:
        yaml.dump(info, f, allow_unicode=True)
    
    print("📋 研究信息已记录:", info_path)
    return info

def insert_cbam_into_model(yolo_model, insert_layers=CBAM_INSERT_LAYERS):
    """
    研究版CBAM插入函数 - 用于算法改进研究
    """
    print(f"🔬 开始CBAM集成研究 - 插入位置: {insert_layers}")
    
    m = yolo_model.model  # DetectionModel
    if not hasattr(m, "model"):
        print("⚠ 模型结构异常，无法进行CBAM集成研究")
        return False

    seq = m.model
    print(f"📊 模型分析: 共{len(seq)}个子模块")

    # 研究用的详细前向传播分析
    device = next(yolo_model.parameters()).device
    yolo_model.eval()

    with torch.no_grad():
        # 使用标准输入尺寸进行网络分析
        dummy_input = torch.zeros((1, 3, 640, 640)).to(device)
        layer_outputs = []
        x = dummy_input

        print("🔍 网络层分析（用于论文技术细节）:")
        for i, layer in enumerate(seq):
            try:
                if hasattr(layer, 'f') and layer.f != -1:
                    if isinstance(layer.f, int):
                        x = layer_outputs[layer.f] if layer.f >= 0 else x
                    else:
                        x = [x if j == -1 else layer_outputs[j] for j in layer.f]

                x = layer(x)
                layer_outputs.append(x)

                # 详细的层分析信息（用于论文）
                if isinstance(x, torch.Tensor):
                    channels = x.shape[1]
                    spatial_size = f"{x.shape[2]}x{x.shape[3]}"
                    print(f"  层{i:2d}: {layer.__class__.__name__:15s} | 通道: {channels:3d} | 空间: {spatial_size:8s}")
                elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                    channels = x[0].shape[1]
                    spatial_size = f"{x[0].shape[2]}x{x[0].shape[3]}"
                    print(f"  层{i:2d}: {layer.__class__.__name__:15s} | 通道: {channels:3d} | 空间: {spatial_size:8s} (多输出)")

            except Exception as e:
                print(f"  层{i:2d}: {layer.__class__.__name__:15s} | 分析失败: {e}")
                layer_outputs.append(None)

    # CBAM集成实验
    print(f"\n🧪 CBAM注意力机制集成实验:")
    inserted = 0
    cbam_info = []
    
    for idx in insert_layers:
        if idx < 0 or idx >= len(seq):
            print(f"❌ 索引{idx}超出范围，跳过")
            continue

        if idx >= len(layer_outputs) or layer_outputs[idx] is None:
            print(f"❌ 索引{idx}无有效输出，跳过")
            continue

        try:
            target = seq[idx]
            output = layer_outputs[idx]

            # 获取通道数
            if isinstance(output, torch.Tensor):
                out_ch = output.shape[1]
                spatial_size = f"{output.shape[2]}x{output.shape[3]}"
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                out_ch = output[0].shape[1]
                spatial_size = f"{output[0].shape[2]}x{output[0].shape[3]}"
            else:
                print(f"❌ 索引{idx}输出格式不支持")
                continue

            # 创建CBAM模块（研究版）
            cbam = CBAM(out_ch, ratio=16, kernel_size=7)  # 标准CBAM配置
            wrapper = CBAMWrapper(target, cbam)
            seq[idx] = wrapper
            
            # 记录CBAM集成信息（用于论文）
            cbam_info.append({
                "layer_index": idx,
                "layer_type": target.__class__.__name__,
                "channels": out_ch,
                "spatial_size": spatial_size,
                "cbam_params": {
                    "ratio": 16,
                    "kernel_size": 7
                }
            })
            
            inserted += 1
            print(f"✅ 层{idx}: {target.__class__.__name__} + CBAM | 通道:{out_ch} | 空间:{spatial_size}")

        except Exception as e:
            print(f"❌ 层{idx}CBAM集成失败: {e}")

    # 保存CBAM集成信息（用于论文分析）
    if cbam_info:
        cbam_info_path = os.path.join(SAVE_DIR, "cbam_integration_analysis.yaml")
        with open(cbam_info_path, "w", encoding='utf-8') as f:
            yaml.dump(cbam_info, f, allow_unicode=True)
        print(f"📊 CBAM集成分析已保存: {cbam_info_path}")

    yolo_model.train()
    print(f"🎯 CBAM集成完成: {inserted}/{len(insert_layers)}个位置成功")
    return inserted > 0

if __name__ == "__main__":
    print("🎓 基于YOLO11+ByteTrack的行人检测与跟踪系统 - 研究训练")
    print("=" * 60)
    
    # 记录研究信息
    research_info = log_research_info()
    
    if USE_CONFIG_FILE:
        print("📋 使用配置文件方式 (适合论文实验)")
        print(f"📄 配置文件: {CBAM_MODEL_YAML}")
        
        # 检查配置文件是否存在
        if not os.path.exists(CBAM_MODEL_YAML):
            print(f"❌ 配置文件不存在: {CBAM_MODEL_YAML}")
            print("💡 请确保已创建CBAM配置文件")
            sys.exit(1)
            
        model = YOLO(CBAM_MODEL_YAML)
        print("✅ 成功加载CBAM增强模型")
        
    else:
        print("🔧 使用动态插入方式 (适合算法研究)")
        model = YOLO(MODEL_PATH)
        print("✅ 成功加载基础YOLO11m模型")
        
        # 进行CBAM集成研究
        cbam_success = insert_cbam_into_model(model, CBAM_INSERT_LAYERS)
        if cbam_success:
            print("🎉 CBAM集成研究成功")
        else:
            print("⚠️ CBAM集成未完全成功，继续使用基础模型")

    # === 研究用超参数配置 ===
    research_hyp = {
        # 学习率策略（适合行人检测）
        "lr0": 0.01,        # 初始学习率
        "lrf": 0.01,        # 最终学习率比例
        "momentum": 0.937,   # SGD动量
        "weight_decay": 0.0005,  # 权重衰减
        
        # 损失函数权重（针对行人检测优化）
        "box": 7.5,         # 边界框损失权重
        "cls": 0.5,         # 分类损失权重
        "dfl": 1.5,         # DFL损失权重
        "fl_gamma": 1.5,    # Focal Loss gamma参数
        
        # 数据增强策略（行人检测专用）
        "mosaic": 1.0,      # Mosaic增强概率
        "mixup": 0.15,      # MixUp增强概率
        "copy_paste": 0.3,  # Copy-Paste增强概率
        
        # 颜色增强
        "hsv_h": 0.015,     # 色调增强
        "hsv_s": 0.7,       # 饱和度增强
        "hsv_v": 0.4,       # 明度增强
        
        # 几何变换
        "degrees": 5.0,     # 旋转角度
        "translate": 0.1,   # 平移比例
        "scale": 0.5,       # 缩放比例
        "shear": 0.0,       # 剪切变换
        "perspective": 0.0, # 透视变换
        "flipud": 0.0,      # 上下翻转
        "fliplr": 0.5,      # 左右翻转
    }

    # 保存研究用超参数
    hyp_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_hyperparameters.yaml")
    with open(hyp_path, "w", encoding='utf-8') as f:
        yaml.dump(research_hyp, f, allow_unicode=True)
    print(f"📋 研究超参数已保存: {hyp_path}")

    # === 开始研究训练 ===
    print("\n🚀 开始研究训练 - 基于YOLO11+CBAM的行人检测")
    print("=" * 60)

    try:
        # 训练配置（适合学术研究）
        training_results = model.train(
            data=DATA_YAML,
            epochs=200,          # 充分训练用于论文实验
            imgsz=640,           # 标准输入尺寸
            batch=16,            # 根据GPU调整
            project=SAVE_DIR,
            name=EXPERIMENT_NAME,
            workers=8,
            optimizer="AdamW",   # 现代优化器
            
            # 直接传入超参数
            lr0=research_hyp["lr0"],
            lrf=research_hyp["lrf"],
            momentum=research_hyp["momentum"],
            weight_decay=research_hyp["weight_decay"],
            box=research_hyp["box"],
            cls=research_hyp["cls"],
            dfl=research_hyp["dfl"],
            fl_gamma=research_hyp["fl_gamma"],
            
            # 数据增强
            mosaic=research_hyp["mosaic"],
            mixup=research_hyp["mixup"],
            copy_paste=research_hyp["copy_paste"],
            hsv_h=research_hyp["hsv_h"],
            hsv_s=research_hyp["hsv_s"],
            hsv_v=research_hyp["hsv_v"],
            degrees=research_hyp["degrees"],
            translate=research_hyp["translate"],
            scale=research_hyp["scale"],
            shear=research_hyp["shear"],
            perspective=research_hyp["perspective"],
            flipud=research_hyp["flipud"],
            fliplr=research_hyp["fliplr"],
            
            # 研究配置
            save=True,           # 保存检查点
            save_period=10,      # 每10轮保存一次
            cache=False,         # 不缓存数据集
            device=0,            # GPU设备
            pretrained=True,     # 使用预训练权重
            verbose=True,        # 详细输出
            
            # 验证配置
            val=True,            # 启用验证
            plots=True,          # 生成训练图表
            
            # 早停配置
            patience=50,         # 早停耐心值
        )

        print("\n🎉 研究训练完成!")
        print(f"📊 训练结果保存在: {SAVE_DIR}")
        print(f"📈 可用于论文的训练图表和日志已生成")
        
        # 保存训练总结（用于论文）
        training_summary = {
            "experiment_info": research_info,
            "training_config": research_hyp,
            "results_path": SAVE_DIR,
            "model_type": "YOLO11m + CBAM" if (USE_CONFIG_FILE or cbam_success) else "YOLO11m",
            "dataset": "Pedestrian Detection",
            "training_epochs": 200,
            "final_metrics": "见训练日志文件"
        }
        
        summary_path = os.path.join(SAVE_DIR, "training_summary_for_paper.yaml")
        with open(summary_path, "w", encoding='utf-8') as f:
            yaml.dump(training_summary, f, allow_unicode=True)
        print(f"📋 论文用训练总结已保存: {summary_path}")

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("💡 建议检查:")
        print("   - 数据集路径和格式")
        print("   - GPU内存是否足够")
        print("   - 调整batch size")
        print("   - 检查CBAM集成是否正确")

    print("\n📚 论文相关文件:")
    print(f"   - 研究信息: {os.path.join(SAVE_DIR, 'research_info.yaml')}")
    print(f"   - 超参数配置: {hyp_path}")
    print(f"   - CBAM分析: {os.path.join(SAVE_DIR, 'cbam_integration_analysis.yaml')}")
    print(f"   - 训练总结: {os.path.join(SAVE_DIR, 'training_summary_for_paper.yaml')}")
    print(f"   - 训练日志和图表: {SAVE_DIR}/{EXPERIMENT_NAME}/")