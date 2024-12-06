import os
import pandas as pd
from collections import defaultdict
import requests
from tqdm import tqdm

numb_of_download = 500

df = pd.read_csv("train_data_yappy/train.csv")
originals = df[df['is_duplicate'] == False]

dataset_folder = 'train_data_yappy/train_dataset'


def find_copies(uuid):
    return df[df['duplicate_for'] == uuid]


def download_file(url, filepath):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        # print(f"Скачано: {filepath}")
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")


# Словарь для хранения оригиналов и их копий
originals_dict = {}

for index, row in originals.iterrows():
    original_uuid = row['uuid']
    copies = find_copies(original_uuid)

    if not copies.empty:
        originals_dict[original_uuid] = copies['uuid'].tolist()

# Словарь для хранения пар, которых нет в train_dataset
missing_files_dict = defaultdict(list)

for original, copies in originals_dict.items():
    original_file = f"{original}.mp4"
    original_path = os.path.join(dataset_folder, original_file)

    original_link = df[df['uuid'] == original]['link'].values[0]
    if not os.path.exists(original_path):
        missing_files_dict[original].append((original, original_link))

    for copy_uuid in copies:
        copy_file = f"{copy_uuid}.mp4"
        copy_path = os.path.join(dataset_folder, copy_file)

        if not os.path.exists(copy_path):
            copy_link = df[df['uuid'] == copy_uuid]['link'].values[0]
            missing_files_dict[original].append((copy_uuid, copy_link))

# Вывод первых 5 недостающих пар
for i, (original, links) in enumerate(missing_files_dict.items()):
    if i >= 5:
        break
    print(
        f"Оригинал: {original}, Ссылка на оригинал: {df[df['uuid'] == original]['link'].values[0]}")
    for copy_uuid, link in links:
        print(f"  Недостающая копия UUID: {copy_uuid}, Ссылка: {link}")
    print("\n")

download_count = 0

for original, links in tqdm(missing_files_dict.items(), desc="Скачивание недостающих файлов", total=min(len(missing_files_dict), numb_of_download)):
    for copy_uuid, link in links:
        filename = os.path.basename(link)
        filepath = os.path.join(dataset_folder, filename)

        download_file(link, filepath)
        download_count += 1

        if download_count >= numb_of_download:
            break

    if download_count >= numb_of_download:
        break
