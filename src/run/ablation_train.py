#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBAM 消融实验脚本
基于 train.py 的现有逻辑，改动最小化

四组实验:
    exp1_baseline      : 无 CBAM（原版 YOLO11）
    exp2_cbam_layer5   : CBAM 只插层 5
    exp3_cbam_layer8   : CBAM 只插层 8
    exp4_cbam_layer5_8 : CBAM 插层 5 + 8（当前方案）

用法（项目根目录下运行）:
    python src/run/ablation_train.py
    python src/run/ablation_train.py --exp exp1_baseline exp4_cbam_layer5_8
    python src/run/ablation_train.py --dry-run
    python src/run/ablation_train.py --epochs 80 --batch 32
"""

import os
import sys
import csv
import argparse
import time
from datetime import datetime
from pathlib import Path

# ── 编码修复（Windows）────────────────────────────────────────────────────────
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ── 路径设置，确保能找到 src/utils/cbam.py ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from utils.cbam import CBAMWrapper, CBAM
except ImportError:
    from src.utils.cbam import CBAMWrapper, CBAM

import torch


# ══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_EXPERIMENTS = {
    "exp1_baseline": {
        "cbam_layers": [],
        "desc": "Baseline — 原版 YOLO11，无 CBAM",
    },
    "exp2_cbam_layer5": {
        "cbam_layers": [5],
        "desc": "CBAM @ Layer5 (P2, 160x160)",
    },
    "exp3_cbam_layer8": {
        "cbam_layers": [8],
        "desc": "CBAM @ Layer8 (P3, 80x80)",
    },
    "exp4_cbam_layer5_8": {
        "cbam_layers": [5, 8],
        "desc": "CBAM @ Layer5 + Layer8（当前方案）",
    },
}

# ── 路径 & 超参（和 train.py 完全一致）───────────────────────────────────────
DATA_YAML = "./dataset/dataset.yaml"
SAVE_DIR  = "result/ablation"
EPOCHS    = 100
BATCH     = 32
IMGSZ     = 640
WORKERS   = 8

POSSIBLE_MODELS = [
    "models/yolo11m.pt", "models/yolo11s.pt", "models/yolo11n.pt",
    "yolo11m.pt", "yolo11s.pt", "yolo11n.pt",
]

TRAIN_HYP = {
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
    "box": 7.5, "cls": 0.5, "dfl": 1.5,
    "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.3,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 5.0, "translate": 0.1, "scale": 0.5, "fliplr": 0.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# CBAM 注入（直接复用 train.py 里的逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def insert_cbam_into_model(yolo_model, insert_layers: list) -> bool:
    """
    完全复用 train.py 里的 insert_cbam_into_model 逻辑。
    insert_layers 为空时不做任何修改（Baseline）。
    """
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

    # 前向推断，获取每层输出尺寸
    with torch.no_grad():
        dummy = torch.zeros((1, 3, IMGSZ, IMGSZ)).to(device)
        layer_outputs = []
        x = dummy
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

    # 插入 CBAM
    inserted = 0
    for idx in insert_layers:
        if idx < 0 or idx >= len(seq):
            print(f"  ❌ 索引 {idx} 超出范围")
            continue
        if idx >= len(layer_outputs) or layer_outputs[idx] is None:
            print(f"  ❌ 索引 {idx} 无有效输出")
            continue

        out = layer_outputs[idx]
        out_ch = out.shape[1] if isinstance(out, torch.Tensor) else out[0].shape[1]

        cbam = CBAM(out_ch, ratio=16, kernel_size=7)
        seq[idx] = CBAMWrapper(seq[idx], cbam)
        inserted += 1
        print(f"  ✅ 层{idx}: channels={out_ch}")

    print(f"  🎯 插入完成: {inserted}/{len(insert_layers)}")
    return inserted > 0


# ══════════════════════════════════════════════════════════════════════════════
# 单次实验
# ══════════════════════════════════════════════════════════════════════════════

def find_model() -> str:
    for p in POSSIBLE_MODELS:
        if os.path.exists(p):
            return p
    return "yolo11m.pt"


def run_experiment(exp_name: str, cfg: dict, dry_run: bool) -> bool:
    print(f"\n{'='*65}")
    print(f"▶  {exp_name}")
    print(f"   {cfg['desc']}")
    print(f"   CBAM 层: {cfg['cbam_layers'] or '无（Baseline）'}")
    print(f"{'='*65}")

    if dry_run:
        print("  [DRY-RUN] 跳过实际训练")
        return True

    try:
        from ultralytics import YOLO

        model_path = find_model()
        print(f"  📁 加载模型: {model_path}")
        model = YOLO(model_path)

        insert_cbam_into_model(model, cfg["cbam_layers"])

        t0 = time.time()
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            optimizer="AdamW",
            project=SAVE_DIR,
            name=exp_name,
            exist_ok=True,
            save=True,
            save_period=20,
            cache="ram",
            patience=30,
            verbose=False,
            val=True,
            plots=True,
            **{k: v for k, v in TRAIN_HYP.items()},
        )

        print(f"\n  ✅ 完成！耗时 {(time.time()-t0)/60:.1f} 分钟")
        return True

    except Exception as e:
        import traceback
        print(f"\n  ❌ 实验失败: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 汇总表格
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(exp_names: list):
    print(f"\n{'='*75}")
    print("📊  消融实验汇总（各实验最优 epoch）")
    print(f"{'='*75}")
    print(f"{'实验':<28} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-'*75}")

    for name in exp_names:
        csv_path = Path(SAVE_DIR) / name / "results.csv"
        if not csv_path.exists():
            print(f"{name:<28} {'(结果文件未找到)':>40}")
            continue

        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue

        best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0))
        m50  = float(best.get("metrics/mAP50(B)",    0))
        m95  = float(best.get("metrics/mAP50-95(B)", 0))
        prec = float(best.get("metrics/precision(B)", 0))
        rec  = float(best.get("metrics/recall(B)",   0))

        print(f"{name:<28} {m50:>8.4f} {m95:>10.4f} {prec:>10.4f} {rec:>8.4f}")

    print(f"{'='*75}")
    print(f"📁 结果目录: {SAVE_DIR}/")
    print("💡 论文建议：以 mAP50 和 mAP50-95 为核心指标，exp4 预期最优")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global EPOCHS, BATCH, DATA_YAML
    parser = argparse.ArgumentParser(description="CBAM 消融实验")
    parser.add_argument("--exp", nargs="+",
                        choices=list(ABLATION_EXPERIMENTS.keys()),
                        default=list(ABLATION_EXPERIMENTS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH)
    parser.add_argument("--data",   default=DATA_YAML)
    args = parser.parse_args()

    
    EPOCHS, BATCH, DATA_YAML = args.epochs, args.batch, args.data

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n🚀  CBAM 消融实验 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   数据集 : {DATA_YAML}")
    print(f"   Epochs : {EPOCHS}  |  Batch: {BATCH}")
    print(f"   输出   : {SAVE_DIR}")
    print(f"   实验组 : {args.exp}")

    results = {}
    for name in args.exp:
        results[name] = run_experiment(name, ABLATION_EXPERIMENTS[name], args.dry_run)

    if not args.dry_run:
        print_summary(args.exp)

    ok = sum(results.values())
    print(f"\n✅  完成 {ok}/{len(results)} 个实验")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"❌  失败: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
