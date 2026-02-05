import subprocess


def call_qwen(image_path, prompt):
    result = subprocess.run(
        [
            "/home/xpekarcik/anaconda3/envs/qwen_env/bin/python",
            "/home/xpekarcik/video-based-action-recognition/djangoweb/source_files/models/run_qwen.py",
            "--image", image_path,
            "--prompt", prompt,
        ],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def call_internvl(image_path, prompt):
    result = subprocess.run(
        [
            "/home/xpekarcik/anaconda3/envs/internvl_env/bin/python",
            "/home/xpekarcik/video-based-action-recognition/djangoweb/source_files/models/run_internvl.py",
            "--image", image_path,
            "--prompt", prompt,
        ],
        capture_output=True,
        text=True
    )
    lines = result.stdout.splitlines()
    cleaned = []

    for line in lines:
        if "FlashAttention2 is not installed" in line :
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()

#not used
def call_paligemma2(image_path, prompt, model_id):
    result = subprocess.run(
        [
            "/home/xpekarcik/anaconda3/envs/paligemma2_env/bin/python",
            "/home/xpekarcik/video-based-action-recognition/djangoweb/source_files/models/run_paligemma2.py",
            "--image", image_path,
            "--prompt", prompt,
            "--model_id", model_id
        ],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()