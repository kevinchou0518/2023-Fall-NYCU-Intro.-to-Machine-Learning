import os
from datasets import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import csv
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
from datasets import load_dataset

def load_images(data_dir):
    image_paths = []
    for image_file in os.listdir(data_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(data_dir, image_file)
            image_paths.append(image_path)
    return image_paths

def create_dataset(data_dir):
    image_paths = load_images(data_dir)
    return Dataset.from_dict({"image_path": image_paths})

# Load the dataset
dataset = create_dataset('./data/test')

# Load the image processor and model
device = "cuda:0" if torch.cuda.is_available() else "cpu"





config = PeftConfig.from_pretrained('finetuned_model/model_lora')
checkpoint = "finetuned_model/model_beit"
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
)
image_processor = AutoImageProcessor.from_pretrained(config.base_model_name_or_path)
# Load the LoRA model
inference_model = PeftModel.from_pretrained(model, 'finetuned_model/model_lora')



normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

transformations = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
    
    ]
)


inference_model = inference_model.to(device)


with open('predictions.csv', mode='w', newline='', encoding='utf-8') as file:

    writer = csv.writer(file)
    writer.writerow(['id', 'label'])  # Writing the header

    # Iterate through the dataset
    for entry in tqdm(dataset):
        # Load the image from the file path
        image = Image.open(entry['image_path']).convert('RGB')
        image = transformations(image)
        # Process the image and predict
        inputs = image_processor(images=image, return_tensors="pt", do_rescale=False, do_normalize=True).to(device)
        with torch.no_grad():
            logits = inference_model(**inputs).logits
        predicted_label = logits.argmax(-1).item()

        # Extract filename without extension from the path
        filename_without_extension, _ = os.path.splitext(os.path.basename(entry['image_path']))

        # Write the filename without extension and the predicted class to the CSV
        predic = predicted_label
        writer.writerow([filename_without_extension, inference_model.config.id2label[predic]])



