from django.apps import AppConfig


class RecognizerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recognizer'
    def ready(self):
        import os

        #only if webserver runs
        if os.environ.get('RUN_MAIN') == 'true':
            from fastapi_services.service_manager import start_services
            start_services()
