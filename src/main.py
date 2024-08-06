import os
import logging
from train import fine_tune_function
from caption import process_images_from_urls
from cloudflare_util import upload
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lora_train(
    image_urls, 
    folder_path, 
    token_name, 
    phrase, 
    unique_id, 
    # pretrained_model_path, 
    output_name, 
    max_train_epochs, 
    learning_rate, 
    unet_lr, 
    text_encoder_lr, 
    network_dim, 
    network_alpha,
    num_repeats,
    lr_warmup_steps,
    lr_scheduler_num_cycles,
    r2_bucket_name,
    r2_access_key_id,
    r2_secret_access_key,
    r2_endpoint_url,
    r2_path_in_bucket,
    ):
    
    
    # Folder Paths
    temp_folder_path = os.path.join(folder_path, unique_id)
    img_folder_path = os.path.join(folder_path, unique_id, "img")
    model_folder_path = os.path.join(temp_folder_path, "model")
    logs_folder_path = os.path.join(temp_folder_path, "logs")
    
    # Lorapath
    lora_path = os.path.join(model_folder_path, f"{output_name}.safetensors")
    
    # base folder
    current_path = os.getcwd()
    
    # Join the current path with the "sd-scripts" folder name
    base_model_path = os.path.join(current_path, 'base')
    pretrained_model_path = os.path.join(base_model_path, 'RealVisXL_V4.0.safetensors')
    
    
    # Process images from URLs and save captions
    logging.info('Starting image captioning...')
    caption_result = process_images_from_urls(image_urls, img_folder_path, token_name, phrase)
    logging.info(f"Captioning result: {caption_result}")

    
    
    # Calculated Values
    save_every_n_epochs = max_train_epochs
    sample_every_n_epochs = 0 # disable it
    train_batch_size = 2

    # Prepare parameters for fine-tuning
    fine_tune_params = {
        "image_dir": img_folder_path,
        "class_tokens": token_name,
        "num_repeats": num_repeats,
        "pretrained_model_path": pretrained_model_path,
        "output_name": output_name,
        "max_train_epochs": max_train_epochs,
        "learning_rate": learning_rate,
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "save_every_n_epochs": save_every_n_epochs,
        "sample_every_n_epochs": sample_every_n_epochs,
        "train_batch_size": train_batch_size,
        "lr_warmup_steps" : lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
        "logs_folder_path": logs_folder_path,
        "model_folder_path": model_folder_path,
        # "r2_bucket_name": r2_bucket_name,
        # "r2_access_key_id": r2_access_key_id,
        # "r2_secret_access_key": r2_secret_access_key,
        # "r2_endpoint_url": r2_endpoint_url,
        # "r2_path_in_bucket": r2_path_in_bucket
    }

    # Start the fine-tuning process
    logging.info('Starting fine-tuning...')
    status = fine_tune_function(fine_tune_params, temp_folder_path)
    logging.info(f"Training status: {status}")

    # Check the status of training
    if status == "success":
        logging.info("Training completed successfully")
        # upload when training is success
        result = upload(
                bucket_name=r2_bucket_name,
                access_key_id=r2_access_key_id,
                secret_access_key=r2_secret_access_key,
                endpoint_url=r2_endpoint_url,
                file_path=lora_path,
                r2_path_in_bucket= r2_path_in_bucket,
                unique_id=unique_id,
                async_upload=False
            )
        
        if result is not None:
            success, message = result
            if success:
                print("Train and Upload was successful!")
            else:
                print(f"Upload failed When Training Was Success: {message}")
        else:
            print("Async upload initiated. Check logs for results.")
        
    else:
        logging.error("Training failed, Nothing Uploaded!")
    
    
    # Clean up temporary folder (loras/id)
    shutil.rmtree(temp_folder_path)
    
    return status