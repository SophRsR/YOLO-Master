from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml")
model.train(
    data='coco.yaml',
    epochs=600,
    batch=256,
    imgsz=640)
model.val(data='coco.yaml')
results = model('../ultralytics/cfg/datasets/coco.yaml') 