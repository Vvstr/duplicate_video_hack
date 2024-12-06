import pytest
import torch
from moviepy.editor import VideoFileClip


def test_audio():
    video_path = 'temp_downloads/00a5f7b0-2d87-4d2a-abbd-0173a9f83c7b.mp4'
    video = VideoFileClip(video_path)
    audio = video.audio

    # Убедитесь, что аудио существует
    assert audio is not None, "Audio track should not be None."

    # Проверка продолжительности аудио
    assert audio.duration > 0, "The audio track has a duration of zero or is invalid."

    # Преобразование в массив звука
    audio_data = audio.to_soundarray(fps=48000)
    assert len(audio_data) > 0, "Audio data should not be empty."


def test_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert device == 'cuda', "Device should be cuda ."

