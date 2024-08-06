import subprocess
import os

def fine_tune_function(params, temp_folder_path):
    # Ensure the temporary folder exists
    os.makedirs(temp_folder_path, exist_ok=True)

    dataset_config = os.path.join(temp_folder_path, "dataset_config.toml")
    train_network_config = os.path.join(temp_folder_path, "train_network_config.toml")
    
    
    # Create dataset_config.toml
    with open(dataset_config, 'w') as f:
        f.write(f"""[[datasets]]
  [[datasets.subsets]]
  image_dir = '{params["image_dir"]}'
  caption_extension = '.txt'
  class_tokens = '{params["class_tokens"]}'
  num_repeats = {params["num_repeats"]}
""")

    # Create train_network_config.toml
    with open(train_network_config, 'w') as f:
        f.write(f"""pretrained_model_name_or_path = "{params["pretrained_model_path"]}"
caption_extension = ".txt"
resolution = "1024,1024"
cache_latents = true
enable_bucket = true
bucket_no_upscale = true
output_dir = "{params["model_folder_path"]}"
output_name = "{params["output_name"]}"
save_precision = "bf16"
save_every_n_epochs = {params["save_every_n_epochs"]}
train_batch_size = {params["train_batch_size"]}
max_token_length = 225
xformers = true
max_train_epochs = {params["max_train_epochs"]}
persistent_data_loader_workers = true
gradient_checkpointing = true
mixed_precision = "bf16"
logging_dir = "{params["logs_folder_path"]}"
sample_every_n_epochs = {params["sample_every_n_epochs"]}
sample_prompts = ""
sample_sampler = "euler_a"
optimizer_type = "AdamW8bit"
learning_rate = {params["learning_rate"]}
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = {params["lr_warmup_steps"]}
lr_scheduler_num_cycles = {params["lr_scheduler_num_cycles"]}
dataset_config = "{dataset_config}"
unet_lr = {params["unet_lr"]}
text_encoder_lr = {params["text_encoder_lr"]}
network_module = "networks.lora"
network_dim = {params["network_dim"]}
network_alpha = {params["network_alpha"]}
""")

    # Get the current working directory
    current_path = os.getcwd()
    
    # Join the current path with the "sd-scripts" folder name
    sd_scripts_path = os.path.join(current_path, 'sd-scripts')
    sdxl_script = os.path.join(sd_scripts_path, 'sdxl_train_network.py')
    
    # Run the training script
    # cmd = f"cd ./sd-scripts && accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py --config_file={train_network_config}"
    
    cmd = f"accelerate launch --num_cpu_threads_per_process=2 {sdxl_script} --config_file={train_network_config}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        status = "success"
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        status = "failed"

    # Clean up temporary files
    # shutil.rmtree(temp_folder_path)

    return status