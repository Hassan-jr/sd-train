import argparse
import logging
from typing import Union, Tuple
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

from library.utils import fire_in_thread, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def get_r2_client(access_key_id: str, secret_access_key: str, endpoint_url: str):
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )

def upload(
    bucket_name: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: str,
    file_path: Union[str, Path],
    r2_path_in_bucket: str,
    unique_id: str,
    async_upload: bool = False
) -> Union[Tuple[bool, str], None]:
    client = get_r2_client(access_key_id, secret_access_key, endpoint_url)

    def uploader() -> Tuple[bool, str]:
        try:
            src_path = Path(file_path)
            if not src_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            object_key = f"{r2_path_in_bucket}/{unique_id}/{src_path.name}"
            
            client.upload_file(str(src_path), bucket_name, object_key)
            logger.info(f"Upload successful: {file_path} -> {bucket_name}/{object_key}")
            return True, "Upload successful"
        except ClientError as e:
            error_msg = f"Failed to upload to R2: {str(e)}"
            if e.response['Error']['Code'] == 'NoSuchBucket':
                error_msg = f"Bucket '{bucket_name}' does not exist. Please create the bucket before uploading."
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to upload to R2: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    if async_upload:
        fire_in_thread(uploader)
        return None
    else:
        return uploader()

def list_objects(
    bucket_name: str,
    prefix: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: str,
):
    client = get_r2_client(access_key_id, secret_access_key, endpoint_url)
    try:
        response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return response.get('Contents', [])
    except Exception as e:
        logger.error(f"Failed to list objects in R2 bucket: {e}")
        return []
