import subprocess
import time
import requests
import atexit
import os
import signal
import sys

class FastAPIServiceManager:
#fastAPI manager - starts qwen, internvl services in the background
    def __init__(self):
        self.services = {
            'qwen': {
                'port': 8001,
                'url': 'http://127.0.0.1:8001/health',
                'script': '/home/xpekarcik/video-based-action-recognition/djangoweb/fastapi_services/qwen_service.py',
                'env': 'qwen_env',
                'python': '/home/xpekarcik/anaconda3/envs/qwen_env/bin/python',
                'process': None,
                'log_file': '/tmp/qwen_service.log'
            },
            'internvl': {
                'port': 8002,
                'url': 'http://127.0.0.1:8002/health',
                'script': '/home/xpekarcik/video-based-action-recognition/djangoweb/fastapi_services/internvl_service.py',
                'env': 'internvl_env',
                'python': '/home/xpekarcik/anaconda3/envs/internvl_env/bin/python',
                'process': None,
                'log_file': '/tmp/internvl_service.log'
            }
        }

    def is_service_running(self, service_name): #check if service is active
        try:
            response = requests.get(
                self.services[service_name]['url'],
                timeout=2
            )
            return response.status_code == 200
        except:
            return False

    def start_service(self, service_name): #start service
        service = self.services[service_name]

        if self.is_service_running(service_name):
            print(f" {service_name.upper()} service already running on port {service['port']}")
            return True

        print(f"Starting {service_name.upper()} service on port {service['port']}...")
        log_file = open(service['log_file'], 'w')

        # Spusti proces na pozadí
        try:
            process = subprocess.Popen(
                [service['python'], service['script']],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Oddelí od parent procesu
            )
        except Exception as e:
            print(f"Failed to start process: {e}")
            log_file.close()
            return False

        service['process'] = process

        #wait 60 seconds for service to start
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

        if service['process']:
            print(f"Stopping {service_name.upper()} service...")
            try:
                #send SIGTERM
                os.killpg(os.getpgid(service['process'].pid), signal.SIGTERM)
                service['process'].wait(timeout=5)
                print(f"{service_name.upper()} service stopped")
            except:
                # if SIGTERM nefunguje, SIGKILL
                try:
                    os.killpg(os.getpgid(service['process'].pid), signal.SIGKILL)
                    print(f"{service_name.upper()} service killed")
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
                print(f"   Django will start anyway, but {service_name} won't work")

        if success:
            print("\n" + "="*60)
            print("All FastAPI services are running!")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("Some services failed to start")
            print("="*60 + "\n")

        return success

    def stop_all(self):
        print("Stopping FastAPI services...")
        for service_name in self.services.keys():
            self.stop_service(service_name)
        print("All services stopped\n")


_manager = None

def get_manager(): #get instance
    global _manager
    if _manager is None:
        _manager = FastAPIServiceManager()
    return _manager

def start_services(): #start all services
    manager = get_manager()
    manager.start_all()
    #Registruj cleanup pri vypnutí Django
    atexit.register(stop_services)

def stop_services(): #stop all services
    manager = get_manager()
    manager.stop_all()