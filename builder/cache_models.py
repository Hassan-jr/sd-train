# builder/model_fetcher.py

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    blip_processor = fetch_pretrained_model(BlipProcessor, "Salesforce/blip-image-captioning-large")
    blip_model = fetch_pretrained_model(BlipForConditionalGeneration, "Salesforce/blip-image-captioning-large").to(device)

    return blip_processor, blip_model


if __name__ == "__main__":
    get_diffusion_pipelines()
