import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import torch

# ✅ Load trained model
model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')
class_names = model.names
num_classes = len(class_names)

# ✅ Paths
test_images_dir = Path('/kaggle/working/datasets/test/images')
test_labels_dir = Path('/kaggle/working/datasets/test/labels')
image_paths = sorted(list(test_images_dir.glob("*.jpg")))

# ✅ Helper to get true labels
def get_true_classes(label_path):
    if not label_path.exists():
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return sorted(list(set([int(line.strip().split()[0]) for line in lines])))

# ✅ Get a randomized collection with 1 image per class (if available)
def get_random_classwise_images():
    classwise_image_dict = {i: [] for i in range(num_classes)}

    random.shuffle(image_paths)  # Shuffle images before looping
    for img_path in image_paths:
        label_path = test_labels_dir / (img_path.stem + ".txt")
        true_classes = get_true_classes(label_path)

        result = model(img_path, verbose=False, conf=0.5)[0]
        pred_classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []

        for cls in set(pred_classes):
            if len(classwise_image_dict[cls]) == 0:
                classwise_image_dict[cls] = (img_path, result, true_classes)
        if all(classwise_image_dict.values()):
            break

    return classwise_image_dict

# ✅ Display a collection
def display_collection(classwise_images, collection_index=1):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"🔍 Randomized Prediction Collection {collection_index}", fontsize=16)

    for i in range(num_classes):
        ax = axes[i // 5][i % 5]
        ax.axis('off')

        if not classwise_images[i]:
            ax.set_title(f"{class_names[i]} (No pred)")
            print(f"[{class_names[i]}] ➤ No prediction found.")
            continue

        img_path, result, true_classes = classwise_images[i]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        # ✅ Find the highest confidence prediction
        if len(confs) > 0:
            top_idx = confs.argmax()
            top_label = class_names[classes[top_idx]]
            top_conf = confs[top_idx]
            print(f"[{class_names[i]}] ➤ Top Pred: {top_label} ({top_conf:.2f}) from {img_path.name}")
        else:
            print(f"[{class_names[i]}] ➤ No predictions in image: {img_path.name}")

        # ✅ Draw all predicted boxes
        for box, cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y2 + 15),  # label at bottom of box
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        true_label_str = ", ".join([class_names[c] for c in true_classes])
        pred_label_str = ", ".join([class_names[c] for c in set(classes)])
        ax.imshow(img)
        ax.set_title(f"T: [{true_label_str}] | P: [{pred_label_str}]", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"visualize_collection_Yolov8customtrain_{collection_index}.png", dpi=300)
    plt.show()

# ✅ Generate & display 3 randomized collections
for i in range(1, 4):
    classwise_images = get_random_classwise_images()
    display_collection(classwise_images, i)
