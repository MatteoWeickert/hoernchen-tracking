from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data="scripts/yolo/dataset/data.yaml",
        epochs=20,
        imgsz=512,
        batch=8,
        device=0
    )

if __name__ == "__main__":
    main()