import streamlit as st
from src.utils import download_file
import torch
import os
import pickle

from src.DB_code import add_embeddings, get_all_data, get_row_by_uuid, create_db, get_audio_embedding_by_uuid
from src.video_analysis import get_video_features
from src.audio_analysis import get_audio_features, load_and_preprocess_audio
from src.video_preprocess import load_and_preprocess_video
from src.utils import get_video_model, get_audio_model, compute_similarity, serialize_and_add_embeddings # find_most_similar_by_video
from src.config import VIDEO_SIMILARITY_THRESHOLD, AUDIO_SIMILARITY_THRESHOLD


def report_video_duplicate(db_uuid, video_similarity, audio_similarity=None):
    if audio_similarity:
        st.markdown(f'<p class="result">Это дубликат видео под ID: {db_uuid},\
            коэффициент сходства по видео равен {video_similarity}, \
                коэффициент сходства по аудио равен {audio_similarity} </p>',
                        unsafe_allow_html=True)
        print(f'Query video is dublicate for uuid = {db_uuid},\
            video_similarity = {video_similarity},\
            audio_similarity = {video_similarity}')
    else:
        st.markdown(f'<p class="result">Это дубликат видео под ID: {db_uuid},\
            коэффициент сходства равен {video_similarity} </p>',
                        unsafe_allow_html=True)
        print(f'Query video is dublicate for uuid = {db_uuid},\
            video_similarity = {video_similarity}')
    
    return True

def report_new_video():
    st.markdown('<p class="result">Похожих видео не найдено.</p>',
                            unsafe_allow_html=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Установка стилей
st.set_page_config(page_title="Сервис распознавания видео", layout="wide")
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4B0082;
    }
    .description {
        font-size: 18px;
        color: #555555;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #008000;
    }
    </style>
""", unsafe_allow_html=True)

# Можно использовать @st.cache_data перед load data которая по идеи нужна для загрузки эмбедингов чтобы сравнить
st.title("Тест сервиса по распознаванию видео", anchor=None)
st.markdown('<p class="description">Введите ссылку на видео для проверки на дубликат.</p>',
            unsafe_allow_html=True)

video_link = st.text_input("Ссылка на видео:")

@st.cache_resource
def load_models():
    video_model = get_video_model(device)
    audio_model = get_audio_model(device)
    return video_model, audio_model

video_model, audio_model = load_models()

create_db()

if st.button("Отправить"):
    video_id = video_link.split('/')[-1].split('.')[0]

    video_path = f'temp_downloads/{video_id}.mp4'

    download_file(video_link, video_path)


    frames = load_and_preprocess_video(video_path)

    if frames is not None:
        query_video_embedding = get_video_features(frames, video_model)
    else:
        raise ValueError('Не удалось загрузить видео :/')

    data = get_all_data()

    db_embeddings = [
        (
            row[0], 
            pickle.loads(row[1]) if row[1] is not None else None, 
            pickle.loads(row[2]) if row[2] is not None else None
        )
        for row in data 
    ]

    query_audio_embedding = None

    flag_duplicate = False
    if db_embeddings:

        for db_uuid, db_video_embedding, db_audio_embedding in db_embeddings:
            video_similarity = compute_similarity(query_video_embedding, db_video_embedding)

            if video_similarity > VIDEO_SIMILARITY_THRESHOLD:
                
                if db_audio_embedding is None:
                    flag_duplicate = report_video_duplicate(db_uuid, 
                                                            video_similarity)
                    break
                else:
                    query_audio_embedding = get_audio_features(model=audio_model, 
                                                            video_path=video_path)

                    if query_audio_embedding is None:
                        flag_duplicate = report_video_duplicate(db_uuid, 
                                                                video_similarity)
                        break

                    else:
                        audio_similarity = compute_similarity(query_audio_embedding, 
                                                            db_audio_embedding)
                        
                        if audio_similarity > AUDIO_SIMILARITY_THRESHOLD:
                            flag_duplicate = report_video_duplicate(db_uuid, 
                                                                    video_similarity, 
                                                                    audio_similarity)
                            break
        
        if not flag_duplicate:          
            report_new_video()
            if not query_audio_embedding:
                query_audio_embedding = get_audio_features(model=audio_model, 
                                                            video_path=video_path)
            serialize_and_add_embeddings(query_video_embedding, query_audio_embedding)

    else:
        report_new_video()
        if not query_audio_embedding:
                query_audio_embedding = get_audio_features(model=audio_model, 
                                                            video_path=video_path)
        serialize_and_add_embeddings(query_video_embedding, query_audio_embedding)
    


    if os.path.exists(video_path):
        os.remove(video_path)