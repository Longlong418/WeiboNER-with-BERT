from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

snapshot_download(repo_id="google-bert/bert-base-chinese",local_dir="./model/bert-base-chinese",resume_download=True,
    ignore_patterns=[
        ".gitattributes",
        "README.md",
        "*.msgpack",
        "*.onnx",
        "*.h5",
        "*.ot",
        "pytorch_model.bin"

    ])#屏蔽其他非pytorch模型以及pytorch_model.bin模型 选择更安全的model.safetensors

snapshot_download(repo_id="hfl/chinese-bert-wwm",local_dir="./model/chinese-bert-wwm",resume_download=True,
    ignore_patterns=[
        ".gitattributes",
        "*.msgpack",
        "README.md",
        "*.h5"
    ])#这个模型没有model.safetensors 所以选择下载pytorch_model.bin