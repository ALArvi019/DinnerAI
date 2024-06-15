import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

class ImageProcessor:
    def __init__(self, model_path='yolov8s.pt'):
        # Initialize the YOLO model from Ultralytics with the specified model path
        self.model = YOLO(model_path)

    def load_image(self, image_path):
        # Load an image from the specified path and convert it to RGB mode
        image = Image.open(image_path).convert("RGB")
        return image

    def load_video(self, video_path):
        # Open a video file for reading using OpenCV
        cap = cv2.VideoCapture(video_path)
        return cap

    def preprocess(self, image):
        # Convert a PIL Image to a numpy array
        image = np.array(image)
        return image

    def detect_objects(self, image):
        # Perform object detection on the input image using the YOLO model
        results = self.model(image)
        return results

    def process_results(self, results):
        # Process the detection results to extract labels and bounding boxes
        labels = []
        boxes = []
        for result in results:
            boxes_data = result.boxes
            class_ids = boxes_data.cls.cpu().numpy()
            for class_id, box in zip(class_ids, boxes_data.xyxy.cpu().numpy()):
                label = self.model.names[int(class_id)]
                labels.append(label)
                boxes.append(box)
        return labels, boxes

    def draw_results(self, image, labels, boxes):
        # Draw bounding boxes and labels on the image based on detection results
        for label, box in zip(labels, boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def display_results(self, labels):
        # Display detected labels
        print("Detected objects:")
        for label in labels:
            print(label)

    def process_image(self, image_path):
        # Process a single image: load, preprocess, detect objects, process results, draw and display
        image = self.load_image(image_path)
        preprocessed_image = self.preprocess(image)
        results = self.detect_objects(preprocessed_image)
        labels, boxes = self.process_results(results)
        self.display_results(labels)
        image_with_results = self.draw_results(preprocessed_image, labels, boxes)
        cv2.imshow("Image", cv2.cvtColor(image_with_results, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_video(self, video_path):
        # Process a video: load, detect objects frame by frame, process results, draw and display
        cap = self.load_video(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detect_objects(image)
            labels, boxes = self.process_results(results)
            self.display_results(labels)
            image_with_results = self.draw_results(image, labels, boxes)
            cv2.imshow("Video", cv2.cvtColor(image_with_results, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()

    # To process a single image
    image_path = "test.jpg"
    processor.process_image(image_path)
    
    # To process a video
    # video_path = "path/to/your/video.mp4"
    # processor.process_video(video_path)
