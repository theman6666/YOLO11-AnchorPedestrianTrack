#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLO11+ByteTrack的行人检测与跟踪系统 - 混合解决方案训练脚本
论文: 基于YOLO11+ByteTrack的行人检测与跟踪系统设计与实现


本脚本自动检测ultralytics源代码完整性，并选择最佳的导入方式：
1. 优先使用本地ultralytics（如果完整且包含我们的CBAM修改）
2. 如果本地不完整，自动切换到系统ultralytics + 动态CBAM插入
"""

import os
import sys
import yaml
import torch

# 设置控制台编码为UTF-8，解决Windows下emoji显示问题
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_local_ultralytics_completeness():
    """检查本地ultralytics源代码是否完整"""
    current_dir = os.getcwd()
    local_ultralytics = os.path.join(current_dir, 'ultralytics')
    
    if not os.path.exists(local_ultralytics):
        return False, "本地ultralytics文件夹不存在"
    
    # 检查关键模块是否存在
    required_modules = [
        'ultralytics/ultralytics/__init__.py',
        'ultralytics/ultralytics/nn/__init__.py',
        'ultralytics/ultralytics/nn/tasks.py',
        'ultralytics/ultralytics/models/__init__.py',
        'ultralytics/ultralytics/engine/__init__.py',
        'ultralytics/ultralytics/data/__init__.py',
        'ultralytics/ultralytics/utils/__init__.py',
    ]
    
    missing_modules = []        # 计算缺失的模块
    for module in required_modules:
        if not os.path.exists(module):
            missing_modules.append(module)
    
    if missing_modules:
        return False, f"缺少关键模块: {missing_modules}"
    
    return True, "本地ultralytics源代码完整"

def setup_ultralytics_import():
    """设置ultralytics导入方式"""
    print("🔍 检查ultralytics源代码完整性...")
    
    is_complete, message = check_local_ultralytics_completeness()
    print(f"📋 检查结果: {message}")
    
    if is_complete:
        print("✅ 使用本地ultralytics源代码（包含CBAM修改）")
        # 使用本地ultralytics
        current_dir = os.getcwd()
        local_ultralytics = os.path.join(current_dir, 'ultralytics')
        if local_ultralytics not in sys.path:
            sys.path.insert(0, local_ultralytics)
        return "local", local_ultralytics
    else:
        print("⚠️ 本地ultralytics不完整，切换到系统ultralytics + 动态CBAM")
        # 使用系统ultralytics
        current_dir = os.getcwd()
        local_ultralytics = os.path.join(current_dir, 'ultralytics')
        
        # 从sys.path中移除本地ultralytics路径
        paths_to_remove = [current_dir, local_ultralytics]
        for path in paths_to_remove:
            while path in sys.path:
                sys.path.remove(path)
        
        # 将当前目录添加到最后
        sys.path.append(current_dir)
        return "system", None

# 设置ultralytics导入
ultralytics_mode, ultralytics_path = setup_ultralytics_import()

try:
    from ultralytics import YOLO
    print(f"✅ 成功导入ultralytics ({ultralytics_mode} 模式)")
    
    # 验证导入的ultralytics版本
    import ultralytics
    actual_path = ultralytics.__file__
    print(f"📍 实际使用的ultralytics路径: {actual_path}")
    
except ImportError as e:
    print(f"❌ 无法导入ultralytics: {e}")
    print("💡 请确保已安装ultralytics包: pip install ultralytics")
    raise

# CBAM utils - 使用本地实现
from src.utils.cbam import CBAMWrapper, CBAM
from src.utils.losses import FocalLoss, siou_loss

# ============= 混合方案配置 =============
DATA_YAML = "./dataset/dataset.yaml"
# 检查多个可能的模型文件，优先使用yolo11s.pt
POSSIBLE_MODELS = [
    "models/yolo11s.pt",  # 使用s版本，更快但准确度没有YOLO11m.pt
    "models/yolo11m.pt",  # 这个的准确率更高，但是没有那么快速
    "models/yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11n.pt"
]
MODEL_PATH = None
for model in POSSIBLE_MODELS:
    if os.path.exists(model):
        MODEL_PATH = model
        print(f"🎯 找到模型文件: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    MODEL_PATH = "yolo11s.pt"  # 默认值，如果不存在会使用YAML创建
    print(f"⚠️ 未找到本地模型文件，将使用: {MODEL_PATH}")

CBAM_MODEL_YAML = "./models/yolo11m_cbam.yaml"
SAVE_DIR = "result/hybrid_weights"
CBAM_INSERT_LAYERS = [5, 8]

# 检查本地模型文件是否存在
def check_local_models():
    """检查本地模型文件可用性"""
    cbam_yaml_exists = os.path.exists(CBAM_MODEL_YAML)
    model_pt_exists = os.path.exists(MODEL_PATH) if MODEL_PATH else False
    
    print(f"📋 本地模型文件检查:")
    print(f"   - CBAM配置文件: {CBAM_MODEL_YAML} {'✅存在' if cbam_yaml_exists else '❌不存在'}")
    if MODEL_PATH:
        print(f"   - 预训练模型: {MODEL_PATH} {'✅存在' if model_pt_exists else '❌不存在'}")
    else:
        print(f"   - 预训练模型: 未找到任何.pt文件")
    
    return cbam_yaml_exists, model_pt_exists

# 检查GPU可用性
def check_gpu_availability():
    """检查GPU和CUDA可用性"""
    import torch
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    
    print(f"🖥️ 硬件检查:")
    print(f"   - CUDA可用: {'✅是' if cuda_available else '❌否'}")
    print(f"   - GPU数量: {device_count}")
    
    if cuda_available:
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   - GPU {i}: {gpu_name}")
        return 0  # 使用第一个GPU
    else:
        print(f"   - 将使用CPU训练")
        return 'cpu'

cbam_yaml_exists, model_pt_exists = check_local_models()
device = check_gpu_availability()

# 根据ultralytics模式和本地文件情况选择策略
if ultralytics_mode == "local" and cbam_yaml_exists:
    USE_CONFIG_FILE = True  # 本地模式且CBAM配置文件存在，优先使用配置文件
    STRATEGY = "本地ultralytics + 配置文件CBAM"
elif cbam_yaml_exists:
    USE_CONFIG_FILE = True  # CBAM配置文件存在，使用配置文件方式
    STRATEGY = "配置文件CBAM模式"
else:
    USE_CONFIG_FILE = False  # 使用动态插入
    STRATEGY = "动态CBAM插入模式"

EXPERIMENT_NAME = f"YOLO11m_CBAM_Hybrid_{ultralytics_mode}"
# =================================

os.makedirs(SAVE_DIR, exist_ok=True)

def insert_cbam_into_model(yolo_model, insert_layers=CBAM_INSERT_LAYERS):
    """
    混合方案的CBAM插入函数
    """
    print(f"🔧 开始动态CBAM插入 (混合方案)")
    
    m = yolo_model.model
    if not hasattr(m, "model"):
        print("⚠ 无法找到模型结构，跳过CBAM插入")
        return False

    seq = m.model
    print(f"📊 模型分析: 共{len(seq)}个子模块，插入位置: {insert_layers}")

    # 获取设备
    device = next(yolo_model.parameters()).device
    yolo_model.eval()

    with torch.no_grad():
        dummy_input = torch.zeros((1, 3, 640, 640)).to(device)
        layer_outputs = []
        x = dummy_input

        print("🔍 逐层分析:")
        for i, layer in enumerate(seq):
            try:
                if hasattr(layer, 'f') and layer.f != -1:
                    if isinstance(layer.f, int):
                        x = layer_outputs[layer.f] if layer.f >= 0 else x
                    else:
                        x = [x if j == -1 else layer_outputs[j] for j in layer.f]

                x = layer(x)
                layer_outputs.append(x)

                if isinstance(x, torch.Tensor):
                    channels = x.shape[1]
                    spatial = f"{x.shape[2]}x{x.shape[3]}"
                    print(f"  层{i:2d}: {layer.__class__.__name__:15s} | 通道:{channels:3d} | 空间:{spatial}")
                elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                    channels = x[0].shape[1]
                    spatial = f"{x[0].shape[2]}x{x[0].shape[3]}"
                    print(f"  层{i:2d}: {layer.__class__.__name__:15s} | 通道:{channels:3d} | 空间:{spatial} (多输出)")

            except Exception as e:
                print(f"  层{i:2d}: {layer.__class__.__name__:15s} | 分析失败: {e}")
                layer_outputs.append(None)

    # CBAM插入
    print(f"\n🧪 CBAM插入实验:")
    inserted = 0
    
    for idx in insert_layers:
        if idx < 0 or idx >= len(seq):
            print(f"❌ 索引{idx}超出范围")
            continue

        if idx >= len(layer_outputs) or layer_outputs[idx] is None:
            print(f"❌ 索引{idx}无有效输出")
            continue

        try:
            target = seq[idx]
            output = layer_outputs[idx]

            if isinstance(output, torch.Tensor):
                out_ch = output.shape[1]
                spatial = f"{output.shape[2]}x{output.shape[3]}"
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                out_ch = output[0].shape[1]
                spatial = f"{output[0].shape[2]}x{output[0].shape[3]}"
            else:
                print(f"❌ 索引{idx}输出格式不支持")
                continue

            # 创建CBAM模块
            cbam = CBAM(out_ch, ratio=16, kernel_size=7)
            wrapper = CBAMWrapper(target, cbam)
            seq[idx] = wrapper
            
            inserted += 1
            print(f"✅ 层{idx}: {target.__class__.__name__} + CBAM | 通道:{out_ch} | 空间:{spatial}")

        except Exception as e:
            print(f"❌ 层{idx}CBAM插入失败: {e}")

    # 不调用train()方法，避免触发数据集检查
    print(f"🎯 CBAM插入完成: {inserted}/{len(insert_layers)}个位置成功")
    return inserted > 0

if __name__ == "__main__":
    print("🎓 基于YOLO11+ByteTrack的行人检测与跟踪系统 - 混合解决方案")
    print("=" * 70)
    print(f"🔧 当前策略: {STRATEGY}")
    print(f"📍 Ultralytics模式: {ultralytics_mode}")
    print("=" * 70)
    
    # 记录混合方案信息
    hybrid_info = {
        "ultralytics_mode": ultralytics_mode,
        "ultralytics_path": ultralytics_path or "系统安装路径",
        "strategy": STRATEGY,
        "use_config_file": USE_CONFIG_FILE,
        "cbam_insert_layers": CBAM_INSERT_LAYERS,
        "experiment_name": EXPERIMENT_NAME
    }
    
    info_path = os.path.join(SAVE_DIR, "hybrid_solution_info.yaml")
    with open(info_path, "w", encoding='utf-8') as f:
        yaml.dump(hybrid_info, f, allow_unicode=True)
    print(f"📋 混合方案信息已保存: {info_path}")

    # 根据模式选择加载方式
    if USE_CONFIG_FILE:
        print(f"📋 使用配置文件方式: {CBAM_MODEL_YAML}")
        
        if not os.path.exists(CBAM_MODEL_YAML):
            print(f"❌ 配置文件不存在: {CBAM_MODEL_YAML}")
            print("🔄 切换到动态插入模式")
            USE_CONFIG_FILE = False
        else:
            try:
                model = YOLO(CBAM_MODEL_YAML)
                print("✅ 成功加载CBAM配置文件模型")
            except Exception as e:
                print(f"❌ 配置文件加载失败: {e}")
                print("🔄 切换到动态插入模式")
                USE_CONFIG_FILE = False
    
    if not USE_CONFIG_FILE:
        print(f"🔧 使用动态插入方式")
        
        # 检查是否有本地预训练模型
        if model_pt_exists:
            print(f"📁 使用本地预训练模型: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
        else:
            print("📁 本地无预训练模型，使用YAML配置创建新模型")
            # 尝试使用本地ultralytics中的配置文件
            local_yaml_options = [
                "./ultralytics/ultralytics/cfg/models/11/yolo11.yaml",  # 基础YOLO11配置
                "yolo11m.yaml",  # 尝试系统路径
                "yolo11.yaml"    # 备用选项
            ]
            
            model_created = False
            for yaml_path in local_yaml_options:
                try:
                    if os.path.exists(yaml_path) or not yaml_path.startswith("./"):
                        print(f"💡 尝试使用配置文件: {yaml_path}")
                        model = YOLO(yaml_path)
                        print(f"✅ 成功从配置文件创建模型: {yaml_path}")
                        model_created = True
                        break
                except Exception as e:
                    print(f"❌ 配置文件 {yaml_path} 创建失败: {e}")
                    continue
            
            if not model_created:
                print("❌ 所有模型创建方式都失败了")
                print("💡 建议:")
                print("   1. 检查网络连接，下载预训练模型")
                print("   2. 或者手动下载 yolo11m.pt 到项目根目录")
                raise RuntimeError("无法创建模型，请检查配置文件或网络连接")
        
        print("✅ 成功加载基础YOLO11m模型")
        
        # 动态插入CBAM
        cbam_success = insert_cbam_into_model(model, CBAM_INSERT_LAYERS)
        if cbam_success:
            print("🎉 动态CBAM插入成功")
        else:
            print("⚠️ CBAM插入未完全成功，继续使用基础模型")

    # === 混合方案超参数 ===
    hybrid_hyp = {
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        # "fl_gamma": 1.5,  # 移除不支持的参数
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
    }

    hyp_path = os.path.join(SAVE_DIR, f"{EXPERIMENT_NAME}_hyperparameters.yaml")
    with open(hyp_path, "w", encoding='utf-8') as f:
        yaml.dump(hybrid_hyp, f, allow_unicode=True)
    print(f"📋 超参数配置已保存: {hyp_path}")

    # === 开始训练 ===
    print(f"\n🚀 开始混合方案训练")
    print("=" * 70)

    try:
        # 根据模型创建方式决定是否使用预训练权重
        use_pretrained = model_pt_exists if not USE_CONFIG_FILE else False
        
        print(f"🎯 训练配置:")
        print(f"   - 使用预训练权重: {'是' if use_pretrained else '否'}")
        print(f"   - 模型类型: {'配置文件模型' if USE_CONFIG_FILE else '预训练模型'}")
        
        model.train(
            data=DATA_YAML,
            epochs=120,
            imgsz=640,
            batch=16,
            project=SAVE_DIR,
            name=EXPERIMENT_NAME,
            workers=8,
            optimizer="AdamW",
            
            # 超参数
            lr0=hybrid_hyp["lr0"],
            lrf=hybrid_hyp["lrf"],
            momentum=hybrid_hyp["momentum"],
            weight_decay=hybrid_hyp["weight_decay"],
            box=hybrid_hyp["box"],
            cls=hybrid_hyp["cls"],
            dfl=hybrid_hyp["dfl"],
            # fl_gamma=hybrid_hyp["fl_gamma"],  # 移除不支持的参数
            
            # 数据增强
            mosaic=hybrid_hyp["mosaic"],
            mixup=hybrid_hyp["mixup"],
            copy_paste=hybrid_hyp["copy_paste"],
            hsv_h=hybrid_hyp["hsv_h"],
            hsv_s=hybrid_hyp["hsv_s"],
            hsv_v=hybrid_hyp["hsv_v"],
            degrees=hybrid_hyp["degrees"],
            translate=hybrid_hyp["translate"],
            scale=hybrid_hyp["scale"],
            fliplr=hybrid_hyp["fliplr"],
            
            # 训练配置
            save=True,
            save_period=10,
            cache=False,
            device=device,  # 自动检测设备
            pretrained=use_pretrained,  # 根据实际情况决定是否使用预训练
            verbose=True,
            val=True,
            plots=True,
            patience=30,
        )

        print("\n🎉 混合方案训练完成!")
        print(f"📊 训练结果保存在: {SAVE_DIR}")
        
        # 保存训练总结
        training_summary = {
            "hybrid_solution": hybrid_info,
            "training_config": hybrid_hyp,
            "results_path": SAVE_DIR,
            "model_enhancement": "CBAM Attention Mechanism",
            "training_status": "完成",
            "notes": f"使用{STRATEGY}成功完成训练"
        }
        
        summary_path = os.path.join(SAVE_DIR, "training_summary.yaml")
        with open(summary_path, "w", encoding='utf-8') as f:
            yaml.dump(training_summary, f, allow_unicode=True)
        print(f"📋 训练总结已保存: {summary_path}")

    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("💡 建议检查:")
        print("   - 数据集路径和格式")
        print("   - GPU内存是否足够")
        print("   - 调整batch size")

    print(f"\n📚 混合方案文件:")
    print(f"   - 方案信息: {info_path}")
    print(f"   - 超参数: {hyp_path}")
    print(f"   - 训练结果: {SAVE_DIR}/{EXPERIMENT_NAME}/")
    # 只有在训练成功时才显示训练总结路径
    if 'summary_path' in locals():
        print(f"   - 训练总结: {summary_path}")