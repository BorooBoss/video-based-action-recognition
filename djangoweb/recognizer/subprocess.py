import requests
import time


def _wait_for_service(url, service_name, max_retries=30):
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            if i == 0:
                print(f"⏳ Waiting for {service_name} service to start...")
            time.sleep(1)
    raise Exception(f"{service_name} service not available after {max_retries}s")


def call_qwen(image_path, prompt): #calls fastAPI server for Qwen
    url = "http://127.0.0.1:8001/predict"
    health_url = "http://127.0.0.1:8001/health"

    #wait for service to start
    _wait_for_service(health_url, "Qwen")

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'prompt': prompt}
            response = requests.post(url, files=files, data=data, timeout=60)

        if response.status_code == 200:
            return response.json()['result']
        else:
            raise Exception(f"Qwen service error: {response.text}")

    except requests.exceptions.ConnectionError:
        raise Exception("Qwen service is not running. Please start Django server first.")
    except requests.exceptions.Timeout:
        raise Exception("Qwen service timeout - model might be overloaded")


def call_internvl(image_path, prompt): #calls fastAPI server for InternVL
    url = "http://127.0.0.1:8002/predict"
    health_url = "http://127.0.0.1:8002/health"

    #wait for service to start
    _wait_for_service(health_url, "InternVL")

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'prompt': prompt}
            response = requests.post(url, files=files, data=data, timeout=60)

        if response.status_code == 200:
            return response.json()['result']
        else:
            raise Exception(f"InternVL service error: {response.text}")

    except requests.exceptions.ConnectionError:
        raise Exception("InternVL service is not running. Please start Django server first.")
    except requests.exceptions.Timeout:
        raise Exception("InternVL service timeout - model might be overloaded")