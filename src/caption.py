import os
import logging
import time
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

def load_blip_model():
    logging.info("Starting to load BLIP model...")
    start_time = time.time()
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        
        end_time = time.time()
        logging.info(f"BLIP model loaded successfully in {end_time - start_time:.2f} seconds")
        return blip_processor, blip_model
    except Exception as e:
        logging.error(f"Error loading BLIP model: {str(e)}")
        raise

# Load models
try:
    blip_processor, blip_model = load_blip_model()
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")
    raise

def generate_blip_caption(image):
    try:
        logging.info(f"Generating BLIP caption for the provided image.")
        inputs = blip_processor(images=image, text="A photo of a man", return_tensors="pt").to(device)
        
        output = blip_model.generate(
            **inputs,
            max_new_tokens=100,
            min_new_tokens=20,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            early_stopping=True
        )
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        logging.info(f"BLIP caption generated successfully.")
        return caption
    except Exception as e:
        logging.error(f"Error in generate_blip_caption: {str(e)}")
        return ""

def format_caption(general_desc, phrase, token_name):
    caption = f"{general_desc}, {phrase}, portrait photography, {token_name}"
    return caption

def download_image(url):
    try:
        logging.info(f"Downloading image from URL: {url}")
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        logging.info(f"Image downloaded successfully from {url}")
        return image
    except Exception as e:
        logging.error(f"Error downloading image from URL {url}: {str(e)}")
        return None

def process_images_from_urls(image_urls, folder_path, token_name, phrase):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, image_url in enumerate(image_urls, start=1):
        try:
            logging.info(f"Processing image {i} from URL: {image_url}")
            
            # Download the image from the URL
            image = download_image(image_url)
            if image is None:
                continue
            
            # Generate BLIP caption
            general_desc = generate_blip_caption(image)
            
            # Format the caption
            formatted_caption = format_caption(
                general_desc,
                phrase,
                token_name
            )
            
            # Save the image and caption
            image_filename = f"{token_name} ({i}).jpg"
            caption_filename = f"{token_name} ({i}).txt"
            
            image_path = os.path.join(folder_path, image_filename)
            caption_path = os.path.join(folder_path, caption_filename)
            
            image.save(image_path)
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(formatted_caption)
            
            logging.info(f"Processed image {i} and saved as {image_filename} and {caption_filename}")
        except Exception as e:
            logging.error(f"Error processing image {i} from URL {image_url}: {str(e)}")
            return json.dumps({"status": "captioning failed"})

    return json.dumps({"status": "Captioning Complete"})

# Example of how to call the function from another script
if __name__ == "__main__":
    image_urls = [
        "https://res.cloudinary.com/zeit-inc/image/upload/nextconf-photos/Sexton_Vercel_1058.jpg",
        "https://res.cloudinary.com/zeit-inc/image/upload/nextconf-photos/Sexton_Vercel_1269.jpg"
    ]
    folder_path = "/content/caption"
    token_name = "r3zman"
    phrase = "A portrait of a man"
    
    result = process_images_from_urls(image_urls, folder_path, token_name, phrase)
    print(result)
