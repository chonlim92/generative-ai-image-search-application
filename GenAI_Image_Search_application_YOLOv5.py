import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model and processor from Hugging Face
clip_model_name = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Load the YOLOv5 model from Hugging Face
yolo_model_name = os.environ.get("YOLO_MODEL_NAME", "yolov5s")
yolo_model = torch.hub.load('ultralytics/yolov5', yolo_model_name, pretrained=True).to(device)

def preprocess_image(image):
    # Convert image mode if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def detect_objects(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    results = yolo_model(image)
    return results

def compare_prompt_with_images(prompt, image_paths):
    # Tokenize the prompt
    inputs = clip_processor(text=[prompt], images=None, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    prompt_embeddings = clip_model.get_text_features(**inputs)
    prompt_embeddings = torch.nn.functional.normalize(prompt_embeddings, p=2, dim=-1)  # Normalize embeddings

    probabilities = []

    for image_path in image_paths:
        # Detect objects in the image using YOLOv5
        results = detect_objects(image_path)
        detected_objects = results.pandas().xyxy[0]

        # Load and preprocess the image for CLIP
        image = Image.open(image_path)
        image = preprocess_image(image)
        inputs = clip_processor(text=None, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
        image_embeddings = clip_model.get_image_features(**inputs)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)  # Normalize embeddings

        # Calculate the cosine similarity between the prompt and image embeddings
        similarity = torch.nn.functional.cosine_similarity(prompt_embeddings, image_embeddings)
        probability = similarity.item() * 100  # Convert to percentage

        probabilities.append((image_path, probability, detected_objects))

    # Sort the results by probability in descending order
    probabilities.sort(key=lambda x: x[1], reverse=True)

    return probabilities

def search_images(prompt, uploaded_files, threshold):
    # Load images from the uploaded files
    image_paths = []
    for uploaded_file in uploaded_files:
        try:
            # Extract the original filename from the uploaded file path
            original_filename = os.path.basename(uploaded_file)
            image_path = os.path.join("temp_images", original_filename)
            os.makedirs("temp_images", exist_ok=True)  # Ensure the directory exists
            Image.open(uploaded_file).save(image_path, format='PNG')  # Save as PNG to ensure compatibility
            image_paths.append(image_path)
        except Exception as e:
            print(f"Alert: Cannot process file '{original_filename}'. Error: {e}")
            continue
    
    if not image_paths:
        return [], []

    # Compare the prompt with the images
    probabilities = compare_prompt_with_images(prompt, image_paths)
    
    # Filter images based on the threshold
    filtered_images = []
    filtered_probs = []
    for image_path, probability, detected_objects in probabilities:
        if probability >= threshold:
            filtered_images.append(image_path)
            filtered_probs.append(probability)
    
    # Debugging: Print the filtered images and their probabilities
    print("Filtered images and their probabilities:")
    for image_path, probability, detected_objects in probabilities:
        if probability >= threshold:
            print(f"File: {image_path}, Probability: {probability}%")
            print(detected_objects)
    
    return filtered_images, [f"{prob:.2f}%" for prob in filtered_probs]

# Create the Gradio interface
def display_results(prompt, uploaded_files, threshold):
    images, probabilities = search_images(prompt, uploaded_files, threshold)
    return images, probabilities

iface = gr.Interface(
    fn=display_results,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your search prompt here..."),
        gr.File(label="Upload Image Files", file_count="multiple", type="filepath"),
        gr.Slider(minimum=1, maximum=100, value=25, label="Probability Threshold (%)")
    ],
    outputs=[
        gr.Gallery(label="Search Results"),
        gr.Textbox(label="Probabilities")
    ],
    title="Image Search using Natural Language"
)

# Launch the Gradio app with share=True to create a public link
if __name__ == "__main__":
    iface.launch(share=True)