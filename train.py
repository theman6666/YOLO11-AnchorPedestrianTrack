import os
import yaml
import torch
from ultralytics import YOLO

# CBAM utils
from src.utils.cbam import CBAMWrapper, CBAM
# optional losses (we will use Focal via hyperparams)
from src.utils.losses import FocalLoss, siou_loss

# ============= 配置 =============
DATA_YAML = "./dataset/your_data.yaml"     # 已生成的 dataset 配置
MODEL_PATH = "./models/YOLO11m.pt"
ANCHOR_PATH = "./models/anchors.yaml"      # 可选（YOLO11 anchor-free 不强制）
SAVE_DIR = "./models/weights_cbam"
CBAM_INSERT_LAYERS = [4, 6]  # 尝试在 backbone 的第 4、6 层后插入 CBAM，若出错可调整
# =================================

os.makedirs(SAVE_DIR, exist_ok=True)

def insert_cbam_into_model(yolo_model, insert_layers=CBAM_INSERT_LAYERS):
    """
    尝试在 yolo_model.model.model 的指定 layer 索引位置将该层封装为 CBAMWrapper。
    这个过程依赖 ultralytics 内部实现（不同版本索引不同），
    所以这里使用 try/except 并打印可调整的建议。
    """
    m = yolo_model.model  # DetectionModel
    # model.model is usually nn.Sequential / list of modules in ultralytics
    if not hasattr(m, "model"):
        print("⚠ 无法找到 m.model，当前 ultralytics 版本结构未知，跳过 CBAM 插入")
        return False

    seq = m.model  # typically nn.Sequential or list
    print(f"模型包含 {len(seq)} 个子模块。尝试插入 CBAM 到索引: {insert_layers}")

    inserted = 0
    for idx in insert_layers:
        if idx < 0 or idx >= len(seq):
            print(f"⚠ 索引 {idx} 超出范围（len={len(seq)})，跳过")
            continue
        try:
            target = seq[idx]
            # try to infer out_channels (a few block types expose .c2 or .conv)
            out_ch = None
            if hasattr(target, "conv"):
                # some blocks have conv attribute
                try:
                    out_ch = int(target.conv.out_channels)
                except Exception:
                    out_ch = None
            elif hasattr(target, "c2"):
                try:
                    out_ch = int(target.c2.out_channels)
                except Exception:
                    out_ch = None
            elif hasattr(target, "m"):
                try:
                    out_ch = int(target.m.out_channels)
                except Exception:
                    out_ch = None

            # fallback: try executing a dummy tensor through to get channel size
            if out_ch is None:
                with torch.no_grad():
                    dummy = torch.zeros((1, 3, 64, 64))
                    out = target(dummy)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_ch = out.shape[1]

            # wrap target module
            wrapper = CBAMWrapper(target, in_channels=out_ch)
            seq[idx] = wrapper
            inserted += 1
            print(f"✅ 在索引 {idx} 插入 CBAM（out_ch={out_ch}）")
        except Exception as e:
            print(f"❌ 无法在索引 {idx} 插入 CBAM：{e}")

    return inserted > 0

if __name__ == "__main__":
    print("🚀 加载 YOLO11m ...")
    model = YOLO(MODEL_PATH)

    # 尝试插入 CBAM（插入失败不会中止训练）
    ok = insert_cbam_into_model(model, CBAM_INSERT_LAYERS)
    if not ok:
        print("⚠ CBAM 插入未成功，请手动调节 CBAM_INSERT_LAYERS 或检查 ultralytics 版本")

    # === 配置超参 / 数据增强 ===
    # 我们将写一个小的 hyp 文件并传入
    hyp = {
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        # loss balance
        "box": 7.5,       # bbox loss weight
        "cls": 0.5,
        "obj": 1.0,
        # focal gamma for classification (ultralytics uses fl_gamma)
        "fl_gamma": 1.5,
        # data augment params (ultralytics 支持某些项)
        "mosaic": 1.0,
        "mixup": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
    }

    # 将 hyp 写入临时文件以备调试/记录
    hyp_path = os.path.join(SAVE_DIR, "hyp_cbam.yaml")
    with open(hyp_path, "w") as f:
        yaml.dump(hyp, f)
    print("🎯 hyperparams saved to:", hyp_path)

    # === 训练（使用 ultralytics 的 API） ===
    print("🔥 开始训练（使用 ultralytics train API），带增强和 focal 分类")
    # 尽量使用 ultralytics 的 train 接口，传入 hyp 参数
    # 注意：ultralytics 接口接受参数名可能随版本变化，这里做最通用调用
    model.train(
        data=DATA_YAML,
        epochs=120,
        imgsz=864,         # 适当放大以提升小目标能力
        batch=8,
        project=SAVE_DIR,
        name="yolo11m_cbam",
        workers=8,
        optimizer="AdamW",
        # ultralytics 允许通过 dict 传入超参（若版本不支持，请手动修改项目 hyp 文件）
        hyp=hyp,
        augment=True,
        pretrained=False,
        device=0
    )

    print("\n🎉 训练完成。权重保存在：", SAVE_DIR)
    print("提示：如果你想把回归 Loss 替换为 SIoU（需要修改 ultralytics 库的 loss 实现或使用自定义训练循环），我可以继续给出补丁方法。")
