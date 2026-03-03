import os
from google.cloud import storage

def download_from_gcs(bucket_name: str, blob_name: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    # If already downloaded in this instance, reuse it
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return dst_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dst_path)
    return dst_path