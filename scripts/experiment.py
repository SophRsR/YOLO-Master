import os
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update(wandb=True)
os.environ["WANDB_PROJECT"] = "yolo-master-reproduce"

model = YOLO("ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml")
model.train(
    data='coco8.yaml',
    epochs=30,
    batch=4,
    imgsz=640,
    device=0)
model.val(data='coco8.yaml')
