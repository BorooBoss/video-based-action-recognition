import subprocess
import time
import requests
import atexit
import os
import signal
import sys

class FastAPIServiceManager:
    # fastAPI manager - starts qwen, internvl services in the background
    def __init__(self):
        self.services = {
            'qwen': {
                'port': 8001,
                'url': 'http://127.0.0.1:8001/health',
                'script': '/home/xpekarcik/video-based-action-recognition/djangoweb/fastapi_services/service_qwen.py',
                'env': 'qwen_env',
                'python': '/home/xpekarcik/anaconda3/envs/qwen_env/bin/python',
                'process': None,
                'log_file': '/tmp/qwen_service.log'
            },
            'internvl': {
                'port': 8002,
                'url': 'http://127.0.0.1:8002/health',
                'script': '/home/xpekarcik/video-based-action-recognition/djangoweb/fastapi_services/service_internvl.py',
                'env': 'internvl_env',
                'python': '/home/xpekarcik/anaconda3/envs/internvl_env/bin/python',
                'process': None,
                'log_file': '/tmp/internvl_service.log'
            }
        }

    def is_service_running(self, service_name):
        try:
            response = requests.get(self.services[service_name]['url'], timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_service(self, service_name):
        service = self.services[service_name]

        if self.is_service_running(service_name):
            print(f" {service_name.upper()} service already running on port {service['port']}")
            return True

        print(f"Starting {service_name.upper()} service on port {service['port']}...")
        log_file = open(service['log_file'], 'w')

        try:
            process = subprocess.Popen(
                [service['python'], service['script']],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
        except Exception as e:
            print(f"Failed to start process: {e}")
            log_file.close()
            return False

        service['process'] = process

        for i in range(60):
            time.sleep(1)
            if process.poll() is not None:
                print(f"{service_name.upper()} process died!")
                log_file.close()
                return False
            if self.is_service_running(service_name):
                print(f"{service_name.upper()} service started successfully!")
                return True
            if i % 5 == 0 and i > 0:
                print(f"Waiting for {service_name} to start... ({i}s)")

        print(f"Failed to start {service_name} service (timeout)")
        log_file.close()
        return False

    def stop_service(self, service_name):
        service = self.services[service_name]
        print(f"Stopping {service_name.upper()} service...")

        # 1. Bleskové zastavenie cez HTTP (ŽIADNE time.sleep!)
        shutdown_url = service['url'].replace('/health', '/shutdown')
        try:
            # Nastavíme super krátky timeout, nechceme čakať, kým sa model fakt vypne.
            # Iba pošleme požiadavku a ideme ďalej.
            requests.post(shutdown_url, timeout=0.5)
            print(f" -> Sent shutdown request to {service_name.upper()} API.")
        except requests.exceptions.ReadTimeout:
            # Ak to nestihne odpovedať do 0.5s, nevadí, request dostal.
            print(f" -> Sent shutdown request to {service_name.upper()} API.")
        except:
            # Ak to hádže iný error, asi už služba beztak nebeží
            pass

        # 2. Fallback: OS Kill (Bez .wait(), čakanie je pre PyCharm smrť)
        if service.get('process'):
            try:
                os.killpg(os.getpgid(service['process'].pid), signal.SIGTERM)
            except:
                try:
                    os.killpg(os.getpgid(service['process'].pid), signal.SIGKILL)
                except:
                    pass
            service['process'] = None

    def start_all(self):
        print("Starting FastAPI services...")
        success = True
        for service_name in self.services.keys():
            if not self.start_service(service_name):
                success = False
                print(f"   Warning: {service_name} service failed to start")

        if success:
            print("\n" + "="*60 + "\nAll FastAPI services are running!\n" + "="*60 + "\n")
        return success

    def stop_all(self):
        print("Stopping FastAPI services...")
        for service_name in self.services.keys():
            self.stop_service(service_name)
        print("All services stopped\n")

# --- Globálne funkcie na správu inštancie ---
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = FastAPIServiceManager()
    return _manager

def start_services():
    manager = get_manager()
    manager.start_all()
    atexit.register(stop_services)

def stop_services():
    manager = get_manager()
    manager.stop_all()