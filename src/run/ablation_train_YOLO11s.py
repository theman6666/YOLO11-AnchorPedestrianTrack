#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBAM 消融实验脚本 - YOLO11s 模型 + 4090D 优化版
支持断点续跑

四组实验:
    exp1_baseline      : 无 CBAM（原版 YOLO11s）
    exp2_cbam_layer5   : CBAM 只插层 5
    exp3_cbam_layer8   : CBAM 只插层 8
    exp4_cbam_layer5_8 : CBAM 插层 5 + 8（当前方案）

用法:
    python src/run/ablation_train.py              # 自动跳过已完成的实验
    python src/run/ablation_train.py --force      # 强制重跑所有实验
    python src/run/ablation_train.py --dry-run    # 仅检查状态不训练
"""

import os, sys, csv, argparse, time
from datetime import datetime
from pathlib import Path

if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from utils.cbam import CBAMWrapper, CBAM
except ImportError:
    from src.utils.cbam import CBAMWrapper, CBAM

import torch

# ══════════════════════════════════════════════════════════════════════════════
ABLATION_EXPERIMENTS = {
    "exp1_baseline":      {"cbam_layers": [],    "desc": "Baseline — 原版 YOLO11s，无 CBAM"},
    "exp2_cbam_layer5":   {"cbam_layers": [5],   "desc": "CBAM @ Layer5 (P2, 160x160)"},
    "exp3_cbam_layer8":   {"cbam_layers": [8],   "desc": "CBAM @ Layer8 (P3, 80x80)"},
    "exp4_cbam_layer5_8": {"cbam_layers": [5, 8],"desc": "CBAM @ Layer5 + Layer8（当前方案）"},
}

DATA_YAML = "./dataset/dataset.yaml"
SAVE_DIR  = "result/ablation_s"  # 修改保存目录，区分s模型
EPOCHS    = 100
BATCH     = 64  # 4090D 显存更大，可以增大batch size
IMGSZ     = 640
WORKERS   = 16  # 增加数据加载线程数

# YOLO11s 模型路径
POSSIBLE_MODELS = [
    "models/yolo11s.pt", 
    "yolo11s.pt",
    "/root/autodl-tmp/yolo11s.pt",  # AutoDL 常用路径
]

# 针对 s 模型优化的训练参数
TRAIN_HYP = {
    "lr0": 0.01,           # 初始学习率
    "lrf": 0.01,           # 最终学习率因子
    "momentum": 0.937,     # SGD动量
    "weight_decay": 0.0005,# 权重衰减
    "warmup_epochs": 3,    # 预热epochs
    "warmup_momentum": 0.8,# 预热动量
    "warmup_bias_lr": 0.1, # 预热偏置学习率
    "box": 7.5,            # box损失增益
    "cls": 0.5,            # cls损失增益
    "dfl": 1.5,            # dfl损失增益
    "mosaic": 1.0,         # Mosaic增强概率
    "mixup": 0.15,         # Mixup增强概率
    "copy_paste": 0.3,     # Copy-paste增强概率
    "hsv_h": 0.015,        # HSV-Hue增强
    "hsv_s": 0.7,          # HSV-Saturation增强
    "hsv_v": 0.4,          # HSV-Value增强
    "degrees": 5.0,        # 旋转角度
    "translate": 0.1,      # 平移
    "scale": 0.5,          # 缩放
    "fliplr": 0.5,         # 水平翻转
}

# 4090D 优化设置
TORCH_COMPILE = False  # 如果使用 torch 2.0+，可以开启编译加速

# ══════════════════════════════════════════════════════════════════════════════
def is_experiment_done(exp_name: str) -> bool:
    """检查实验是否已有有效结果（best.pt存在 + results.csv有足够数据）"""
    exp_dir  = Path(SAVE_DIR) / exp_name
    best_pt  = exp_dir / "weights" / "best.pt"
    csv_path = exp_dir / "results.csv"

    if not best_pt.exists() or not csv_path.exists():
        return False
    try:
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        return len(rows) >= 5
    except Exception:
        return False


def insert_cbam_into_model(yolo_model, insert_layers: list) -> bool:
    """插入 CBAM 模块"""
    if not insert_layers:
        print("  ℹ️  insert_layers 为空，保持原始模型（Baseline）")
        return True

    print(f"  🔧 动态 CBAM 插入，目标层: {insert_layers}")
    m = yolo_model.model
    if not hasattr(m, "model"):
        print("  ⚠️  无法找到模型结构，跳过")
        return False

    seq = m.model
    device = next(yolo_model.parameters()).device
    yolo_model.eval()

    # 获取每层输出通道数
    with torch.no_grad():
        dummy = torch.zeros((1, 3, IMGSZ, IMGSZ)).to(device)
        layer_outputs, x = [], dummy
        for i, layer in enumerate(seq):
            try:
                if hasattr(layer, 'f') and layer.f != -1:
                    if isinstance(layer.f, int):
                        x = layer_outputs[layer.f] if layer.f >= 0 else x
                    else:
                        x = [x if j == -1 else layer_outputs[j] for j in layer.f]
                x = layer(x)
                layer_outputs.append(x)
            except Exception as e:
                print(f"  层{i:2d} 前向失败: {e}")
                layer_outputs.append(None)

    inserted = 0
    for idx in insert_layers:
        if idx < 0 or idx >= len(seq):
            print(f"  ❌ 索引 {idx} 超出范围"); continue
        if idx >= len(layer_outputs) or layer_outputs[idx] is None:
            print(f"  ❌ 索引 {idx} 无有效输出"); continue
        out = layer_outputs[idx]
        out_ch = out.shape[1] if isinstance(out, torch.Tensor) else out[0].shape[1]
        
        # 对于s模型，使用稍大的ratio以平衡参数量
        seq[idx] = CBAMWrapper(seq[idx], CBAM(out_ch, ratio=16, kernel_size=7))
        inserted += 1
        print(f"  ✅ 层{idx}: channels={out_ch}")

    print(f"  🎯 插入完成: {inserted}/{len(insert_layers)}")
    return inserted > 0


def find_model() -> str:
    """查找模型文件"""
    for p in POSSIBLE_MODELS:
        if os.path.exists(p):
            print(f"  📁 找到模型: {p}")
            return p
    print("  ⚠️  未找到本地模型，将尝试自动下载 yolo11s.pt")
    return "yolo11s.pt"  # 让 ultralytics 自动下载


def get_gpu_info():
    """获取 GPU 信息并显示"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🎮 GPU: {gpu_name} | 显存: {gpu_mem:.1f} GB")
        return gpu_name, gpu_mem
    return None, None


def run_experiment(exp_name: str, cfg: dict, dry_run: bool, force: bool) -> bool:
    """运行单个实验"""
    print(f"\n{'='*65}")
    print(f"▶  {exp_name}")
    print(f"   {cfg['desc']}")
    print(f"   CBAM 层: {cfg['cbam_layers'] or '无（Baseline）'}")
    print(f"{'='*65}")

    # ── 跳过已完成实验 ────────────────────────────────────────────────────────
    if not force and is_experiment_done(exp_name):
        print(f"  ⏭️  已有完整结果，跳过（使用 --force 可强制重跑）")
        return True

    if dry_run:
        status = "✅ 已完成" if is_experiment_done(exp_name) else "⏳ 待训练"
        print(f"  [DRY-RUN] 状态: {status}")
        return True

    try:
        from ultralytics import YOLO
        
        # 获取 GPU 信息
        get_gpu_info()
        
        # 加载模型
        model_path = find_model()
        print(f"  📁 加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 插入 CBAM
        insert_cbam_into_model(model, cfg["cbam_layers"])

        # 训练配置
        train_args = {
            "data": DATA_YAML,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "workers": WORKERS,
            "optimizer": "AdamW",
            "project": SAVE_DIR,
            "name": exp_name,
            "exist_ok": True,
            "save": True,
            "save_period": 20,
            "cache": "ram",
            "patience": 30,
            "verbose": False,
            "val": True,
            "plots": True,
            "device": 0,  # 使用 GPU 0
            "amp": True,  # 启用混合精度训练
            "fraction": 1.0,  # 使用全部数据
            "profile": False,
            **TRAIN_HYP,
        }
        
        # 如果使用 torch 2.0+，可以尝试编译
        if TORCH_COMPILE:
            try:
                import torch._dynamo
                train_args["torch_compile"] = True
                print("  ⚡ 启用 torch.compile 加速")
            except:
                pass

        # 开始训练
        t0 = time.time()
        print(f"  🚀 开始训练... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        
        model.train(**train_args)
        
        # 训练完成
        elapsed = (time.time() - t0) / 60
        print(f"\n  ✅ 完成！耗时 {elapsed:.1f} 分钟")
        
        # 验证最终模型
        print(f"  🔍 验证最终模型...")
        val_results = model.val()
        print(f"     mAP50: {val_results.box.map50:.4f}, mAP50-95: {val_results.box.map:.4f}")
        
        return True

    except Exception as e:
        import traceback
        print(f"\n  ❌ 实验失败: {e}")
        traceback.print_exc()
        return False


def print_summary(exp_names: list):
    """打印实验结果汇总"""
    print(f"\n{'='*75}")
    print("📊  消融实验汇总（YOLO11s）")
    print(f"{'='*75}")
    print(f"{'实验':<28} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-'*75}")

    best_results = {}
    for name in exp_names:
        csv_path = Path(SAVE_DIR) / name / "results.csv"
        if not csv_path.exists():
            print(f"{name:<28} {'(结果文件未找到)':>40}")
            continue
        try:
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            
            # 找到最佳 epoch (按 mAP50)
            best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0))
            m50  = float(best.get("metrics/mAP50(B)",    0))
            m95  = float(best.get("metrics/mAP50-95(B)", 0))
            prec = float(best.get("metrics/precision(B)", 0))
            rec  = float(best.get("metrics/recall(B)",   0))
            
            best_results[name] = {
                "mAP50": m50, "mAP50-95": m95, 
                "precision": prec, "recall": rec,
                "epoch": best.get("epoch", "?")
            }
            
            print(f"{name:<28} {m50:>8.4f} {m95:>10.4f} {prec:>10.4f} {rec:>8.4f}")
        except Exception as e:
            print(f"{name:<28} {'(读取失败)':>40}")

    print(f"{'='*75}")
    
    # 如果有基线结果，计算提升
    if "exp1_baseline" in best_results:
        print("\n📈  相对基线的提升：")
        baseline = best_results["exp1_baseline"]
        for name in exp_names:
            if name == "exp1_baseline" or name not in best_results:
                continue
            imp_m50 = best_results[name]["mAP50"] - baseline["mAP50"]
            imp_m95 = best_results[name]["mAP50-95"] - baseline["mAP50-95"]
            print(f"  {name:<20}: mAP50 {imp_m50:+.4f}, mAP50-95 {imp_m95:+.4f}")
    
    print(f"\n📁 结果目录: {SAVE_DIR}/")


def main():
    global EPOCHS, BATCH, DATA_YAML, WORKERS
    parser = argparse.ArgumentParser(description="CBAM 消融实验 - YOLO11s")
    parser.add_argument("--exp", nargs="+", choices=list(ABLATION_EXPERIMENTS.keys()),
                        default=list(ABLATION_EXPERIMENTS.keys()))
    parser.add_argument("--dry-run", action="store_true", help="仅检查状态，不训练")
    parser.add_argument("--force",   action="store_true", help="强制重跑，忽略已有结果")
    parser.add_argument("--epochs",  type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--batch",   type=int, default=BATCH, help="批次大小")
    parser.add_argument("--workers", type=int, default=WORKERS, help="数据加载线程数")
    parser.add_argument("--data",    default=DATA_YAML, help="数据集配置文件")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile 加速")
    args = parser.parse_args()

    # 更新全局变量
    EPOCHS, BATCH, DATA_YAML, WORKERS = args.epochs, args.batch, args.data, args.workers
    global TORCH_COMPILE
    TORCH_COMPILE = args.compile
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 打印实验信息
    print(f"\n{'='*75}")
    print(f"🚀  CBAM 消融实验 - YOLO11s")
    print(f"{'='*75}")
    print(f"   时间    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   数据集  : {DATA_YAML}")
    print(f"   Epochs  : {EPOCHS}")
    print(f"   Batch   : {BATCH}")
    print(f"   Workers : {WORKERS}")
    print(f"   输出    : {SAVE_DIR}")
    print(f"   实验组  : {args.exp}")
    
    # GPU 信息
    gpu_name, gpu_mem = get_gpu_info()
    
    # 显存估计
    est_mem = (BATCH * IMGSZ * IMGSZ * 3 * 4 * 2) / (1024**3) * 10  # 粗略估计
    if gpu_mem and est_mem > gpu_mem * 0.9:
        print(f"  ⚠️  估计显存使用 {est_mem:.1f}GB，接近 GPU 上限 {gpu_mem:.1f}GB")
        print("     建议减小 batch size 或 imgsz")
    
    if args.force:
        print("   ⚠️  --force 模式：忽略已有结果，全部重跑")
    if args.compile:
        print("   ⚡ --compile 模式：启用 torch.compile 加速")
    print(f"{'='*75}")

    # 运行实验
    results = {}
    for name in args.exp:
        results[name] = run_experiment(
            name, ABLATION_EXPERIMENTS[name], args.dry_run, args.force)

    # 打印汇总
    if not args.dry_run:
        print_summary(args.exp)

    # 统计结果
    ok = sum(results.values())
    print(f"\n✅  完成 {ok}/{len(results)} 个实验")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"❌  失败: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()