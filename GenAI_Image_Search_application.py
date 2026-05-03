import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.")

import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import base64
import os
import csv
from datetime import datetime
import io
from dotenv import load_dotenv

load_dotenv()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model and processor from Hugging Face
model_name = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

print(f"Task is running on : {device}")

def preprocess_image(image):
    # Convert image mode if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def search_images(prompt, uploaded_files, threshold):
    # Load images from the uploaded files
    images = []
    image_paths = []
    for uploaded_file in uploaded_files:
        try:
            # Extract the original filename from the uploaded file path
            original_filename = os.path.basename(uploaded_file)
            image = Image.open(uploaded_file)
            image = preprocess_image(image)
            images.append(image)
            # Save image data to temporary files
            image_path = os.path.join("temp_images", original_filename)
            os.makedirs("temp_images", exist_ok=True)  # Ensure the directory exists
            image.save(image_path, format='PNG')  # Save as PNG to ensure compatibility
            image_paths.append(image_path)
        except Exception as e:
            print(f"Alert: Cannot process file '{original_filename}'. Error: {e}")
            continue
    
    if not images:
        return [], []

    # Process the images and the prompt
    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    
    # Move the inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get the image and text embeddings
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=0)  # Change dim to 0 to get probabilities across images
    
    # Debugging: Print the probabilities
    print("Probabilities:", probs)
    
    # Get the indices of the images sorted by relevance
    sorted_indices = torch.argsort(probs.squeeze(), descending=True).tolist()
    
    # Ensure sorted_indices is a list even if there's only one image
    if isinstance(sorted_indices, int):
        sorted_indices = [sorted_indices]
    
    # Filter images based on the threshold
    filtered_images = []
    filtered_probs = []
    for i in sorted_indices:
        if i < len(image_paths):  # Ensure the index is within the valid range
            probability = probs.squeeze()[i].item() * 100  # Convert to percentage
            if probability >= threshold:
                filtered_images.append(image_paths[i])
                filtered_probs.append(probability)
    
    # Debugging: Print the filtered images and their probabilities
    print("Filtered images and their probabilities:")
    for i in sorted_indices:
        if i < len(image_paths):  # Ensure the index is within the valid range
            probability = probs.squeeze()[i].item() * 100
            if probability >= threshold:
                print(f"File: {image_paths[i]}, Probability: {probability}%")
    
    return filtered_images, filtered_probs

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def display_uploaded_images(uploaded_files):
    if not uploaded_files:
        return []
    thumbnails = []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            image = preprocess_image(image)
            image.thumbnail((300, 300))  # Resize to 300x300 pixels
            thumbnail_path = os.path.join("temp_thumbnails", os.path.basename(uploaded_file))
            os.makedirs("temp_thumbnails", exist_ok=True)  # Ensure the directory exists
            image.save(thumbnail_path, format='PNG')
            thumbnails.append(thumbnail_path)
        except Exception as e:
            print(f"Alert: Cannot create thumbnail for file '{uploaded_file}'. Error: {e}")
            continue
    return thumbnails

def export_results_as_csv(prompt, filtered_images, filtered_probs):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time}_{prompt}_imagemining_result.csv"
    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')
    writer.writerow(["Index", "Image File Name", "Probability"])
    for index, (image_path, probability) in enumerate(zip(filtered_images, filtered_probs)):
        image_filename = os.path.basename(image_path)  # Extract the filename from the path
        writer.writerow([index, image_filename, probability])
    
    # Ensure the temp_csv directory exists
    os.makedirs("temp_csv", exist_ok=True)
    filepath = os.path.join("temp_csv", filename)
    
    return filepath, output.getvalue()

def export_csv(prompt):
    global filtered_images_list, probabilities_list
    filename, csv_content = export_results_as_csv(prompt, filtered_images_list, probabilities_list)
    with open(filename, "w") as f:
        f.write(csv_content)
    return filename

# Create the Gradio Blocks interface
title = "Prototype: GenAI Natural Language Image Mining"

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (!url.searchParams.has('__theme')) {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Variable to store probabilities
probabilities_list = []
filtered_images_list = []

with gr.Blocks(js=js_func, title=title) as genaiagent:

    gr.Markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <h1 style="font-size: 75px; margin-left: 10px;">{title}</h1>
        </div>

        <br> 

        <p> 
            <strong>Instruction: </strong>
            <br><br>
            1. Upload all raw image files in the Upload Image Files canvas. These image files will be used for mining. 
            <br>
            2. Input a natural language prompt in the Prompt for Image Mining canvas, which will be used for the image mining.
            <br>
            3. Drag and set a probability threshold. Images with the mining probability less than this threshold will be filtered out.
            <br>
            4. The Uploaded raw image files can be previewed as thumbnails in the Uploaded Images canvas.
            <br>
            5. Press on the "Start Image Mining" button to trigger the image mining process. The process duration is depending on the amount of uploaded raw image files, as well as the availability of the GPU.
            <br>
            6. Once the mining process is completed, the fulfilled images will be displayed on the Mining Results canvas. Feel free to select, zoom or download any specific image from the result.
            <br>
            7. The probabilites of all the result images are displayed in the Probability Panel. If a specific image is selected in the Mining Results canvas above, only the probability of that specific image will be shown.
               If no image is selected, the whole probability list will be shown. 
            <br>
            8. Press on the "Export Result as CSV" button to download the mining result as a csv with the columns [index, image file name, probability].
            <br>
            9. Once the csv file is ready to be downloaded, its link will be listed in the Download CSV canvas. You can then click on the link to download the csv file.
        </p>

        <p>
            <strong><span style="color: red;">DISCLAIMER: The mining results might still contain some False Positives and False Negatives. Manual review on the result is strongly recommended. </span></strong>
        </p>

        """)

    with gr.Row():
        with gr.Column(scale=1):
            uploaded_files = gr.File(label="Upload Image Files", file_count="multiple", type="filepath", file_types=["image"])
            prompt = gr.Textbox(label="Prompt for Image Mining", lines=2, placeholder="Enter your mining prompt here...")
            threshold = gr.Slider(minimum=1, maximum=100, value=25, label="Probability Threshold (%)")
            start_button = gr.Button("Start Image Mining")
            thumbnails = gr.Gallery(label="Uploaded Images")
        with gr.Column(scale=2):
            search_results = gr.Gallery(label="Mining Results")
            probabilities = gr.Textbox(label="Probability")
            export_button = gr.Button("Export Result as CSV")
            download_link = gr.File(label="Download CSV", visible=True)  # Make the download link visible

    def display_results(prompt, uploaded_files, threshold):
        global probabilities_list, filtered_images_list
        images, probabilities = search_images(prompt, uploaded_files, threshold)
        probabilities_list = probabilities  # Update the global variable
        filtered_images_list = images  # Update the global variable
        return images, ", ".join([f'{i:.2f}%' for i in probabilities_list])

    def update_thumbnails(uploaded_files):
        return display_uploaded_images(uploaded_files)

    def get_select_value(evt: gr.SelectData):
        global probabilities_list
        print(f"Selected image: {evt.value['image']['orig_name']}")
        return f"{probabilities_list[evt.index]:.2f}%"

    def show_all_probabilities():
        global probabilities_list
        return ", ".join([f'{i:.2f}%' for i in probabilities_list])

    def trigger_download(prompt):
        filename = export_csv(prompt)
        return filename

    uploaded_files.change(update_thumbnails, uploaded_files, thumbnails)
    start_button.click(display_results, [prompt, uploaded_files, threshold], [search_results, probabilities])
    search_results.select(get_select_value, None, probabilities)
    search_results.preview_close(show_all_probabilities, None, probabilities)
    export_button.click(trigger_download, prompt, download_link)

if __name__ == "__main__":
    genaiagent.queue(max_size=1).launch(share=True, debug=True)