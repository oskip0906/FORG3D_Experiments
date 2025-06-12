from huggingface_hub import snapshot_download

local_dir_path = "qwen2-vl-2b-instruct"

local_dir = snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir=local_dir_path,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Downloaded to:", local_dir)