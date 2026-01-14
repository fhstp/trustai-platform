import os
import requests
import time
from dotenv import load_dotenv
from tqdm import tqdm

# === LOAD ENVIRONMENT VARIABLES ===
load_dotenv()  # reads the .env file in the current directory

LABEL_STUDIO_URL = os.getenv("API_BASE_URL")
API_TOKEN = os.getenv("API_TOKEN")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_API_URL = os.getenv("ML_BACKEND_URL")

if not all([LABEL_STUDIO_URL, API_TOKEN, PROJECT_ID, MODEL_API_URL]):
    raise ValueError("❌ Missing required environment variables in .env file.")

headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json",
}

def get_project_details():
    """Fetch project info to retrieve parsed_label_config."""
    resp = requests.get(f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}", headers=headers)
    resp.raise_for_status()
    project = resp.json()
    parsed_label_config = project.get("parsed_label_config")
    if not parsed_label_config:
        raise ValueError("❌ parsed_label_config not found in project response.")
    print("✅ Retrieved parsed_label_config from Label Studio.")
    return parsed_label_config

def get_all_tasks():
    """Retrieve all tasks from the Label Studio project."""
    print("Fetching tasks from Label Studio...")
    tasks = []
    page = 1
    while True:
        try:
            resp = requests.get(
                f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks?page={page}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            tasks.extend(data)
            page += 1
        except requests.exceptions.HTTPError:
            break
    print(f"✅ Retrieved {len(tasks)} tasks.")
    return tasks


def get_predictions_from_backend(tasks, parsed_label_config):
    """Send each task's data to the model backend and collect predictions."""
    print("Requesting predictions from model backend...")

    # Send the task data to your model backend
    payload = {
        "tasks": tasks,
        "project": int(PROJECT_ID),
        "parsed_label_config": parsed_label_config,
    }
    model_response = requests.post(f"{MODEL_API_URL}/predict", json=payload)
    model_response.raise_for_status()
    predictions = model_response.json()
    print(predictions['results'])

    print(f"✅ Collected {len(predictions['results'])} predictions.")
    return predictions['results']


def upload_predictions(predictions):
    for pred in predictions:
        resp = requests.post(
            f"{LABEL_STUDIO_URL}/api/predictions",
            headers=headers,
            json=pred,
        )
        if resp.status_code not in (200, 201):
            print(f"❌ Failed for task {task_id}: {resp.status_code} {resp.text}")
        else:
            print(f"✅ Uploaded prediction for task {pred['task']}")


if __name__ == "__main__":
    parsed_label_config = get_project_details()
    tasks = get_all_tasks()
    predictions = get_predictions_from_backend(tasks, parsed_label_config)
    if predictions:
        upload_predictions(predictions)
    else:
        print("⚠️ No predictions to upload.")

