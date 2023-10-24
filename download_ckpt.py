import urllib.request
from tqdm import tqdm

def download_checkpoint(url, save_path):
    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as file:
            file_size = int(response.info().get('Content-Length', -1))
            chunk_size = 8192
            num_chunks = file_size // chunk_size if file_size > chunk_size else 1

            with tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading', ncols=100) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), b''):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Checkpoint downloaded and saved to: {save_path}")
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")

if __name__ == "__main__":
    download_checkpoint("https://huggingface.co/One-2-3-45/code/resolve/main/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth")

