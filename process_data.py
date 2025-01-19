import os
import shutil
import torchaudio

def get_audio_duration(file_path):
    """Получает длительность аудиофайла."""
    waveform, sample_rate = torchaudio.load(file_path)
    duration_in_seconds = waveform.size(1) / sample_rate
    return duration_in_seconds

def get_subset_of_data(data_path, target_hours=15):
    """Собирает подмножество аудиофайлов и их транскрипций для заданной длительности."""
    audio_files = []
    total_duration = 0.0

    for root, _, files in os.walk(data_path):
        # Ищем текущий `.trans.txt` файл
        trans_file = None
        for file in files:
            if file.endswith(".trans.txt"):
                trans_file = os.path.join(root, file)
                break

        # Если файл с транскрипциями найден, загружаем их
        if trans_file:
            transcriptions = {}
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    audio_id, text = line.strip().split(" ", 1)
                    transcriptions[audio_id] = text

            # Проходимся по аудиофайлам в этой папке
            for file in files:
                if file.endswith(".flac"):
                    audio_id = file.replace(".flac", "")
                    if audio_id in transcriptions:
                        audio_path = os.path.join(root, file)
                        duration = get_audio_duration(audio_path)
                        audio_files.append((audio_path, audio_id, transcriptions[audio_id], duration))
                        total_duration += duration

    # Сортировка и выбор подмножества
    audio_files.sort(key=lambda x: x[3], reverse=True)  # Сортируем по длительности
    selected_files = []
    selected_duration = 0.0
    for audio_file, audio_id, text, duration in audio_files:
        if selected_duration + duration <= target_hours * 3600:  # 3600 секунд в часе
            selected_files.append((audio_file, audio_id, text))
            selected_duration += duration

    return selected_files

def save_subset(data_path, selected_files, output_path):
    """Сохраняет выбранные аудиофайлы и транскрипции в заданную структуру."""
    os.makedirs(output_path, exist_ok=True)

    trans_dict = {}  # Словарь для группировки транскрипций по подкаталогам
    for audio_file, audio_id, text in selected_files:
        # Копируем аудиофайл
        relative_audio_path = os.path.relpath(audio_file, data_path)
        output_audio_path = os.path.join(output_path, relative_audio_path)
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        shutil.copy(audio_file, output_audio_path)

        # Готовим транскрипцию для соответствующего каталога
        dir_key = os.path.dirname(output_audio_path)
        if dir_key not in trans_dict:
            trans_dict[dir_key] = []
        trans_dict[dir_key].append(f"{audio_id} {text}")

    # Записываем транскрипции в соответствующие `.trans.txt` файлы
    for dir_path, transcriptions in trans_dict.items():
        # Создаем правильное имя для транскрипционного файла
        relative_dir = os.path.relpath(dir_path, output_path)
        parts = relative_dir.split(os.sep)
        trans_file_name = "-".join(parts) + ".trans.txt"  # Пример: "19-198.trans.txt"
        trans_file_path = os.path.join(dir_path, trans_file_name)

        # Сохраняем транскрипции
        with open(trans_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcriptions))

# Пример использования
data_path = "LibriSpeech/train-clean-100"  # Путь к вашему набору данных
output_path = "./LibriSpeech/train-clean-15"  # Папка, куда будут сохранены обрезанные данные

# Получение списка файлов, которые составляют 25 часов
selected_files = get_subset_of_data(data_path, target_hours=15)

# Сохранение нового поднабора данных
save_subset(data_path, selected_files, output_path)

print(f"Created subset with {len(selected_files)} files, totaling approximately 15 hours.")
