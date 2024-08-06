import torch
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schemas import INPUT_SCHEMA
from main import lora_train 

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #

class ModelHandler:
    def __init__(self):
        self.load_models()

    def load_models(self):
        pass  # Since your generate_image function downloads models, this is not needed

MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #

@torch.inference_mode()
def generate_image_handler(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    # Call your generate_image function
    result = lora_train(
        image_urls=job_input['image_urls'],
        folder_path=job_input['folder_path'],
        token_name=job_input['token_name'],
        phrase=job_input['phrase'],
        unique_id=job_input['unique_id'],
        output_name=job_input['output_name'],
        max_train_epochs=job_input['max_train_epochs'],
        learning_rate=job_input['learning_rate'],
        unet_lr=job_input['unet_lr'],
        text_encoder_lr=job_input['text_encoder_lr'],
        network_dim=job_input['network_dim'],
        network_alpha=job_input['network_alpha'],
        num_repeats=job_input['num_repeats'],
        lr_warmup_steps=job_input['lr_warmup_steps'],
        lr_scheduler_num_cycles=job_input['lr_scheduler_num_cycles'],
        r2_bucket_name=job_input['r2_bucket_name'],
        r2_access_key_id=job_input['r2_access_key_id'],
        r2_secret_access_key=job_input['r2_secret_access_key'],
        r2_endpoint_url=job_input['r2_endpoint_url'],
        r2_path_in_bucket=job_input['r2_path_in_bucket']
    )
    return result

runpod.serverless.start({"handler": generate_image_handler})
 