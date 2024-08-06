INPUT_SCHEMA = {
    'image_urls': {
        'type': list,
        'required': True,
        'default': [] # images to train
    },
    'folder_path': {
        'type': str,
        'required': False,
        'default': "loras" # temporary folder for storing training result like model
    },
    'token_name': {
        'type': str,
        'required': True #train token like r3zman
    },
    'phrase': {
        'type': str,
        'required': True # a young man
    },
    'unique_id': {
        'type': str,
        'required': True # user id
    },
    'output_name': {
        'type': str,
        'required': True # character name
    },
    'max_train_epochs': {
        'type': int,
        'required': False,
        'default': 15
    },
    'learning_rate': {
        'type': float,
        'required': False,
        'default': 0.0001
    },
    'unet_lr': {
        'type': float,
        'required': False,
        'default': 0.0001
    },
    'text_encoder_lr': {
        'type': float,
        'required': False,
        'default': 5e-5
    },
    'network_dim': {
        'type': int,
        'required': False,
        'default': 8
    },
    'network_alpha': {
        'type': int,
        'required': False,
        'default': 1
    },
    'num_repeats': {
        'type': int,
        'required': False,
        'default': 30
    },
    'lr_warmup_steps': {
        'type': int,
        'required': False,
        'default': 500
    },
    'lr_scheduler_num_cycles': {
        'type': int,
        'required': False,
        'default': 3
    },
    'r2_bucket_name': {
        'type': str,
        'required': True
    },
    'r2_access_key_id': {
        'type': str,
        'required': True
    },
    'r2_secret_access_key': {
        'type': str,
        'required': True
    },
    'r2_endpoint_url': {
        'type': str,
        'required': True
    },
    'r2_path_in_bucket': {
        'type': str,
        'required': False,
        'default': "Loras"
    },
}
