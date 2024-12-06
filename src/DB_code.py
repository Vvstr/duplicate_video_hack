import sqlite3
import os
from .config import DB_PATH


def create_db():
    directory = os.path.dirname(DB_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    conn = sqlite3.connect(os.path.join(directory, "hack_embeddings.db"))
    c = conn.cursor()
    c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings(
               uuid INTEGER PRIMARY KEY AUTOINCREMENT,
               embedding_video BLOB,
               embedding_audio BLOB
            )
    ''')

    conn.commit()
    conn.close()


def add_embeddings(embedding_video, embedding_audio):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (embedding_video, embedding_audio) VALUES (?, ?)',
              (embedding_video, embedding_audio))
    conn.commit()
    conn.close()


def get_all_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT uuid, embedding_video, embedding_audio FROM embeddings')
    data = c.fetchall()
    conn.close()
    return data

def get_row_by_uuid(uuid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT uuid, embedding_video, embedding_audio FROM embeddings WHERE uuid = ?', (uuid,))
    data = c.fetchall()
    conn.close()
    return data

def get_audio_embedding_by_uuid(uuid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT embedding_audio FROM embeddings WHERE uuid = ?', (uuid,))
    
    data = c.fetchone()
    
    conn.close()
    
    return data[0] if data is not None else None  
