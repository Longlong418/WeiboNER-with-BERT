from huggingface_hub import snapshot_download
from Config import Config
from tqdm import tqdm

config=Config()

for repo in tqdm(config.models.keys()):
    model_dir = snapshot_download(repo_id=repo, cache_dir=config.cache_dir)
    print(f"{repo} 下载完成，保存在: {model_dir}")