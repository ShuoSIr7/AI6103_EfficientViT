import os
import subprocess
import zipfile


def download_dataset(url, output_path):
    print(f"Downloading dataset from {url}...")
    subprocess.run(["wget", "-O", output_path, url], check=True)
    print(f"Dataset downloaded to {output_path}")

# 解压数据集
def unzip_dataset(zip_path, extract_to):
    print(f"Unzipping dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Dataset unzipped to {extract_to}")

if __name__ == "__main__":

    output_dir = "./Imagenet100"
    data_url = "https://huggingface.co/datasets/lostboiii6/ImageNet100/resolve/main/ImageNet100.zip"
    zip_path = os.path.join(output_dir, "dataset.zip")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 下载和解压数据集
    try:
        download_dataset(data_url, zip_path)
        unzip_dataset(zip_path, output_dir)
        os.remove(zip_path)
        print("Temporary ZIP file removed. Download complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

