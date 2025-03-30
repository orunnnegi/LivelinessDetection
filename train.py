<<<<<<< HEAD
from ultralytics import YOLO

# Set the device to 'cpu' explicitly
model = YOLO('yolov8n.pt')

def main():
    # Reduce the batch size
    # Set your desired batch size
    model.train(data='C:/Liveliness Detection/Dataset/SplitData/data.yaml', epochs= 50, imgsz=640, batch=2, device='cpu', workers=2)

if __name__ == '__main__':
    main()
=======
from ultralytics import YOLO

# Set the device to 'cpu' explicitly
model = YOLO('yolov8n.pt')

def main():
    # Reduce the batch size
    # Set your desired batch size
    model.train(data='C:/Liveliness Detection/Dataset/SplitData/data.yaml', epochs= 50, imgsz=640, batch=2, device='cpu', workers=2)

if __name__ == '__main__':
    main()
>>>>>>> 0058476 (Initial commit of Liveliness Detection project)
