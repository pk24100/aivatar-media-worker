import requests
import time

ENDPOINT_ID = ""
API_KEY = ""
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_offline_inference():
    """Test basic offline inference through RunPod endpoint."""

    payload = {
        "input": {
            "mode": "offline",
            "sourceImage": "C:\\Users\\prajw\\Pictures\\Screenshots\\keerthi-suresh.jpg",
            "audioPath": "C:\\Users\\prajw\\Pictures\\Screenshots\\sad_dialogue.mp3"
        }
    }

    print("Submitting job to RunPod...")
    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    job_id = data["id"]
    print(f"Job submitted: {job_id}")

    # Poll for completion
    while True:
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        status_resp.raise_for_status()
        status_data = status_resp.json()

        print(f"Status: {status_data.get('status')} - {status_data.get('message', '')}")

        if status_data["status"] == "COMPLETED":
            print("SUCCESS! Output:", status_data.get("output"))
            return True
        elif status_data["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print("FAILED! Error:", status_data.get("error"))
            return False

        time.sleep(5)

if __name__ == "__main__":
    success = test_offline_inference()
    exit(0 if success else 1)
