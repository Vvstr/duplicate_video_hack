import pickle
import os


def save_obj(obj, filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    try:
        with open(f'{filepath}.pkl', 'wb') as file:
            pickle.dump(obj, file)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        return False


def read_obj(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return False
