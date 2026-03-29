import os
import subprocess

DITTO_REPO_PATH = os.getenv("DITTO_REPO_PATH", "/app/ditto-talkinghead")

def run_ditto_inference(model_root, audio_path, source_image, output_mp4):
    """
    Calls official Ditto inference script.
    Uses the converted TensorRT engines if available, else ONNX.
    """

    # best practice: pick TRT if exists (fastest)
    trt_path = os.path.join(model_root, "ditto_trt_3090")
    cfg_pkl  = os.path.join(model_root, "ditto_cfg/v0.4_hubert_cfg_trt.pkl")

    # if no TRT dir present, fallback to onnx
    if not os.path.isdir(trt_path) or len(os.listdir(trt_path)) == 0:
        trt_path = os.path.join(model_root, "ditto_onnx")

    # call the official inference script
    inference_script = os.path.join(DITTO_REPO_PATH, "inference.py")
    cmd = [
        "python3", inference_script,
        "--data_root", trt_path,
        "--cfg_pkl", cfg_pkl,
        "--audio_path", audio_path,
        "--source_path", source_image,
        "--output_path", output_mp4
    ]
    print("Running Ditto command:", cmd)
    subprocess.run(cmd, check=True)
    return output_mp4
