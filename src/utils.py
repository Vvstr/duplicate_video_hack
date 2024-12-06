import os
import requests
import laion_clap
import pickle

from models.model import SimilarityRecognizer
from .DB_code import add_embeddings
from .config import VIDEO_MODEL_LOCAL_PATH, VIDEO_MODEL_URL, AUDIO_MODEL_LOCAL_PATH, AUDIO_MODEL_URL


def download_file(object_url, download_path):
    directory = os.path.dirname(download_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    response = requests.get(object_url)

    if response.status_code == 200:
        with open(download_path, 'wb') as file:
            file.write(response.content)
    else:
        print("Ошибка при скачивании файла:", response.status_code)


def compute_similarity(q_feat, d_feat, topk_cs=True):
    sim = q_feat @ d_feat.T
    sim = sim.max(dim=1)[0]
    if topk_cs:
        sim = sim.sort()[0][-3:]
    sim = sim.mean().item()
    return sim


def get_video_model(device):
    if not os.path.exists(VIDEO_MODEL_LOCAL_PATH):
        download_file(VIDEO_MODEL_URL, VIDEO_MODEL_LOCAL_PATH)
    video_model = SimilarityRecognizer(model_type="base", batch_size=8)
    video_model.to(device)
    video_model.load_pretrained_weights(
        "checkpoints/best_model_base_224_16x16_rgb.pth")
    video_model.eval()

    return video_model


def get_audio_model(device):
    if not os.path.exists(AUDIO_MODEL_LOCAL_PATH):
        download_file(AUDIO_MODEL_URL, AUDIO_MODEL_LOCAL_PATH)
    audio_model = laion_clap.CLAP_Module(
        enable_fusion=False, device=device, amodel='HTSAT-base')
    audio_model.load_ckpt(
        'checkpoints/music_speech_audioset_epoch_15_esc_89.98.pt')
    audio_model.eval()

    return audio_model

def serialize_and_add_embeddings(video_embedding=None, audio_embedding=None):
    if video_embedding is not None:
        video_embedding = pickle.dumps(video_embedding)
    if audio_embedding is not None:
        audio_embedding = pickle.dumps(audio_embedding)

    add_embeddings(video_embedding, audio_embedding)




