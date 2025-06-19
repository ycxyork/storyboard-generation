from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
    local_dir="model/alimama-creative",
    local_dir_use_symlinks=False
)
