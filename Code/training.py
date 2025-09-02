import os
import yaml
import torch
from ultralytics import YOLO
from cbam_enhancement import get_dynamic_yaml

# ✅ 10. Training Settings
base_channels = 64
epochs = 20
batch = 16
lr = 0.0005
optimizer = "Adam"
imgsz = 640
variants = ['n']  # All variants

# ✅ 11. Training Loop
if __name__ == '__main__':
    for v in variants:
        print(f"\n🔧 Preparing YOLOv8{v.upper()}...")

        # Get the YAML content from the cbam_enhancement module
        yaml_content = get_dynamic_yaml(base_channels, variant=v)
        yaml_file = f"yolov8{v}-cbam.yaml"

        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        model_CBAM = YOLO(yaml_file)

        # Optional AMP patching if model has a trainer
        if hasattr(model_CBAM, 'trainer') and hasattr(model_CBAM.trainer, 'args'):
            model_CBAM.trainer.args.amp = False

        #wandb.init(project="YOLOv8_CBAM_AllVariants", name=f"YOLOv8{v.upper()}_CBAM")

        print(f"🚀 Training YOLOv8{v.upper()} CBAM...")
        model_CBAM.train(
            data="data.yaml",  # ✅ Ensure your data.yaml is in place
            epochs=epochs,
            imgsz=imgsz,
            optimizer=optimizer,
            lr0=lr,
            batch=batch,
            amp=False,
            deterministic=False,

            # Augmentations
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=5.0,
            translate=0.05,
            scale=0.1,
            shear=0.5,
            perspective=0.0,
            flipud=0.1,
            fliplr=0.3,
            mosaic=0.3,
            mixup=0.05,
            copy_paste=0.05,
        )

        #wandb.finish()
        print(f"✅ Finished training YOLOv8{v.upper()}.\n")
