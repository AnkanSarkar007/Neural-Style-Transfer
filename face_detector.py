import os, sys
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

# Set the device for running the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the MTCNN module
mtcnn = MTCNN(keep_all=True, device=device)

# Define a transform to normalize the image data
transform = transforms.Compose([transforms.ToTensor()])

def detect_faces(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(img_rgb)

    # Draw bounding boxes on the image
    if boxes is not None:
        for box in boxes:
            cv2.rectangle(img, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          (0, 255, 0), 
                          2)

    return img, boxes

def process_folder(folder_path, output_folder):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                print(file)
                image_path = os.path.join(root, file)

                # Detect faces in the image
                _, face_boxes = detect_faces(image_path)

                # Save the entire image only if faces are detected and the face is a dominant element
                if face_boxes is not None:
                    img = cv2.imread(image_path)

                    # Calculate the area of the image
                    img_area = img.shape[0] * img.shape[1]

                    for i, box in enumerate(face_boxes):
                        # Calculate the area of the detected face bounding box
                        face_area = (box[2] - box[0]) * (box[3] - box[1])

                        # If the face area covers a significant portion of the image, consider it a portrait
                        if face_area > 0.15 * img_area:
                            cv2.imwrite(os.path.join(output_folder, file), img)
                            break  # Break if at least one face is considered a portrait
                    
if __name__ == "__main__":
    style = sys.argv[1]
    folder_path = f"wikiart/{style}"
    output_folder = f"output_{style}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(device)
    process_folder(folder_path, output_folder)
