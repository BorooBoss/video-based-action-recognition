from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recognize/', views.recognize, name='recognize'),
    path('convert_video/', views.convert_video, name='convert_video'),
    path('instructions/', views.instructions_page, name='instructions'),

    # Video frames endpoints
    path('video_frames/', views.video_frames, name='video_frames'),
    path('frame/<str:filename>', views.get_frame, name='get_frame'),
    path('clear_frames/', views.clear_frames, name='clear_frames'),
]
