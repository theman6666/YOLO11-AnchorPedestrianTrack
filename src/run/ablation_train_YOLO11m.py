#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBAM 消融实验脚本 - 支持断点续跑

四组实验:
    exp1_baseline      : 无 CBAM（原版 YOLO11）
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
    "exp1_baseline":      {"cbam_layers": [],    "desc": "Baseline — 原版 YOLO11，无 CBAM"},
    "exp2_cbam_layer5":   {"cbam_layers": [5],   "desc": "CBAM @ Layer5 (P2, 160x160)"},
    "exp3_cbam_layer8":   {"cbam_layers": [8],   "desc": "CBAM @ Layer8 (P3, 80x80)"},
    "exp4_cbam_layer5_8": {"cbam_layers": [5, 8],"desc": "CBAM @ Layer5 + Layer8（当前方案）"},
}

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
        seq[idx] = CBAMWrapper(seq[idx], CBAM(out_ch, ratio=16, kernel_size=7))
        inserted += 1
        print(f"  ✅ 层{idx}: channels={out_ch}")

    print(f"  🎯 插入完成: {inserted}/{len(insert_layers)}")
    return inserted > 0


def find_model() -> str:
    for p in POSSIBLE_MODELS:
        if os.path.exists(p):
            return p
    return "yolo11m.pt"


def run_experiment(exp_name: str, cfg: dict, dry_run: bool, force: bool) -> bool:
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
        model_path = find_model()
        print(f"  📁 加载模型: {model_path}")
        model = YOLO(model_path)
        insert_cbam_into_model(model, cfg["cbam_layers"])

        t0 = time.time()
        model.train(
            data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            workers=WORKERS, optimizer="AdamW", project=SAVE_DIR,
            name=exp_name, exist_ok=True, save=True, save_period=20,
            cache="ram", patience=30, verbose=False, val=True, plots=True,
            **TRAIN_HYP,
        )
        print(f"\n  ✅ 完成！耗时 {(time.time()-t0)/60:.1f} 分钟")
        return True

    except Exception as e:
        import traceback
        print(f"\n  ❌ 实验失败: {e}")
        traceback.print_exc()
        return False


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


def main():
    global EPOCHS, BATCH, DATA_YAML
    parser = argparse.ArgumentParser(description="CBAM 消融实验（支持断点续跑）")
    parser.add_argument("--exp", nargs="+", choices=list(ABLATION_EXPERIMENTS.keys()),
                        default=list(ABLATION_EXPERIMENTS.keys()))
    parser.add_argument("--dry-run", action="store_true", help="仅检查状态，不训练")
    parser.add_argument("--force",   action="store_true", help="强制重跑，忽略已有结果")
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--batch",   type=int, default=BATCH)
    parser.add_argument("--data",    default=DATA_YAML)
    args = parser.parse_args()

    EPOCHS, BATCH, DATA_YAML = args.epochs, args.batch, args.data
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n🚀  CBAM 消融实验 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   数据集 : {DATA_YAML}")
    print(f"   Epochs : {EPOCHS}  |  Batch: {BATCH}")
    print(f"   输出   : {SAVE_DIR}")
    print(f"   实验组 : {args.exp}")
    if args.force:
        print("   ⚠️  --force 模式：忽略已有结果，全部重跑")

    results = {}
    for name in args.exp:
        results[name] = run_experiment(
            name, ABLATION_EXPERIMENTS[name], args.dry_run, args.force)

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