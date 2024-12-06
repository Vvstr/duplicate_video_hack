import torch
from .video_preprocess import Preprocess


def get_video_features(frames, model):
    """Функция, в которой выделяется признаки видео с помощью модели"""

    with torch.no_grad():
        feats = model.extract_features(frames.cuda())

    feats = feats.detach().cpu()

    normed_feats = model.normalize_features(feats)

    return normed_feats
