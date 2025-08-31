from kvmm.models import yolo

model = yolo.YoloV5s(input_shape=(640, 640, 3), weights=None)