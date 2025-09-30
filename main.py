import sys
import json
import copy
from pathlib import Path
import pandas as pd
import argparse
import logging
import importlib
import torch
import tensorflow as tf

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Определение путей проекта
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# Добавление корневой директории в sys.path для импортов
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pipeline_mod = importlib.import_module("src.pipeline")
TimeSeriesPipeline = getattr(pipeline_mod, "TimeSeriesPipeline")


def find_data_folders(base_dir: str) -> list:
    """
    Ищет датасеты: в каждой подпапке base_dir
    проверяет наличие файлов requests.csv.zstd и errors.csv.zstd.
    Args:
        base_dir: Путь к директории, внутри которой будут
            просканированы  подпапки
    Returns:
        list: список путей к подпапкам, в каждой
            из которых найдены оба файла: "requests.csv.zstd" и "errors.csv.zstd".
            Если подходящих подпапок нет или директория не существует, то
            возвращается пустой список.
    """
    base_path = Path(base_dir).resolve()
    folders = []

    print(f"CWD: {Path.cwd().resolve()}", flush=True)
    print(f"Scan dir: {base_path}", flush=True)

    if not base_path.exists():
        print(f"Directory doesnt exist: {base_dir}", flush=True)
        return folders

    found = 0
    for folder in sorted(p for p in base_path.iterdir() if p.is_dir()):
        req = folder / "requests.csv.zstd"
        err = folder / "errors.csv.zstd"

        if req.exists() and err.exists():
            print(f"Found dataset: {folder.name}", flush=True)
            folders.append(folder)
            found += 1
        else:
            print(f"Skip: {folder.name}", flush=True)

    if found == 0:
        print("No matching subfolders were found", flush=True)

    print(f"Number of datasets: {len(folders)}", flush=True)
    return folders


def update_config_for_folder(config_template: dict, folder_path: Path) -> dict:
    """Обновляет конфиг для конкретной папки, подстявляя путь folder_path
    Args:
        config_template: Шаблон конфигурации, который будет скопирован
        folder_path: Путь к папке датасета, внутри которой ожидаются
            файлы requests.csv.zstd и errors.csv.zstd
    Returns:
        dict: Новая глубокая копия конфигурации с обновлёнными путями в
        config['data']['requests_file'] и config['data']['errors_file'],
        указывающими на файлы внутри folder_path.
    """
    config = copy.deepcopy(config_template)
    config['data']['requests_file'] = str(folder_path / "requests.csv.zstd")
    config['data']['errors_file'] = str(folder_path / "errors.csv.zstd")

    return config


def collect_daily_metrics_from_folder(results_dir: Path, folder_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Собирает daily_metrics из файлов, созданных pipeline в указанной папке
    разделяет на requests и errors датафреймы.

    Args:
        results_dir: директория с результатами для папки
        folder_name: имя папки (для логирования)

    Returns:
        (requests_daily_df, errors_daily_df)
    """
    daily_files = list(results_dir.glob("*_daily.csv"))

    if not daily_files:
        print(f"No daily files in {results_dir}", flush=True)
        return pd.DataFrame(), pd.DataFrame()

    requests_daily = []
    errors_daily = []

    for daily_file in daily_files:
        try:
            daily_df = pd.read_csv(daily_file)
            daily_df['source_file'] = daily_file.name
            daily_df['folder'] = folder_name

            # Определяем тип по имени файла
            filename = daily_file.name.lower()
            if 'request' in filename:
                requests_daily.append(daily_df)
                print(f"Load REQUESTS daily file: {daily_file.name}", flush=True)
            elif 'error' in filename:
                errors_daily.append(daily_df)
                print(f"Load ERRORS daily file: {daily_file.name}", flush=True)
            else:
                # Если не можем определить тип то добавим в оба
                print(f"Unknown daily file type: {daily_file.name}, add to both", flush=True)
                requests_daily.append(daily_df)
                errors_daily.append(daily_df)

        except Exception as e:
            print(f"Failed to load {daily_file}: {e}", flush=True)

    # Объединяем requests метрики
    if requests_daily:
        combined_requests = pd.concat(requests_daily, ignore_index=True)
        print(f"Merged {len(requests_daily)} REQUESTS daily files for {folder_name}", flush=True)
    else:
        combined_requests = pd.DataFrame()

    if errors_daily:
        combined_errors = pd.concat(errors_daily, ignore_index=True)
        print(f"Merged {len(errors_daily)} ERRORS daily files for {folder_name}", flush=True)
    else:
        combined_errors = pd.DataFrame()

    return combined_requests, combined_errors


def run_pipeline_for_folder(folder_path: Path, config_template: dict, use_decoder: bool = False,
                            single_dataset: str = None, tcn: bool = False,
                            deepar: bool = False) -> dict:
    """
    Запускает пайплайн для одной папки датасета и возвращает результаты

    Args:
        folder_path: Путь к папке
        config_template:  конфиг модели
        use_decoder: Использовать DecoderMLP
        single_dataset: возможность прогнать модель по 'requests' или 'errors' если  None — оба
        tcn: Использовать TCN
        deepar: Использовать  DeepAR

    Returns:
        dict: Результаты выполнения со сводной информацией

    """
    print(f"\n{'=' * 60}", flush=True)
    print(f" Processing folder: {folder_path.name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    model_type = "DeepAR" if deepar else ("TCN" if tcn else (
        "DecoderMLP" if use_decoder else "LSTM"))
    try:
        # добавляем в конфиг пути к файлам
        config = update_config_for_folder(config_template, folder_path)

        results_dir = PROJECT_ROOT / "results" / folder_path.name
        results_dir.mkdir(parents=True, exist_ok=True)

        config['output']['results_dir'] = str(results_dir)
        config['output']['save_results'] = True
        config['output']['save_daily_metrics'] = True

        pipeline = TimeSeriesPipeline(config, use_decoder_mlp=use_decoder,
                                      use_deepar=deepar, use_tcn=tcn)

        print(f"Launching {model_type} models for {folder_path.name}", flush=True)

        if single_dataset:
            print(f"Single-dataset mode: {single_dataset}", flush=True)
            results = pipeline.compare_single_dataset(single_dataset)
            requests_daily = pd.DataFrame()
            errors_daily = pd.DataFrame()
            if single_dataset == 'requests':
                requests_daily, _ = collect_daily_metrics_from_folder(results_dir, folder_path.name)
            elif single_dataset == 'errors':
                _, errors_daily = collect_daily_metrics_from_folder(results_dir, folder_path.name)
        else:
            if tcn or deepar:
                results = pipeline.compare_architectures_multi()
            else:
                results = pipeline.compare_architectures_single()
            requests_daily, errors_daily = collect_daily_metrics_from_folder(results_dir, folder_path.name)

        print(f"\nResults for {folder_path.name}:", flush=True)
        print(results.to_string(index=False), flush=True)

        return {
            'folder': folder_path.name,
            'success': True,
            'results': results,
            'model_type': model_type,
            'raw_requests_daily': requests_daily,
            'raw_errors_daily': errors_daily,
            'results_dir': str(results_dir),
            'single_dataset': single_dataset or 'both'
        }

    except Exception as e:
        print(f" ERROR while processing {folder_path.name}: {e}", flush=True)
        import traceback
        traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'folder': folder_path.name,
            'success': False,
            'error': str(e),
            'model_type': model_type,
            'single_dataset': single_dataset or 'both'
        }


def main():
    parser = argparse.ArgumentParser(description='пакетная обработка временных рядов')
    parser.add_argument('--data-dir', '-d', default='data/folder_with_data', help='Папка с подпапками данных')
    parser.add_argument('--config', '-c', default='config/config_lstm.json', help='Файл конфигурации')
    parser.add_argument('--decoder', action='store_true', help='Использовать DecoderMLP вместо LSTM')
    parser.add_argument('--tcn', action='store_true', help='Использовать TCN')
    parser.add_argument('--deepar', action='store_true', help='Использовать DEEPAR')
    parser.add_argument('--gpu-check', action='store_true', help='проверить GPU')
    parser.add_argument('--single-dataset', choices=['requests', 'esrrors'],
                        help='Обработать только один тип датасета (requests или errors)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Data directory: {args.data_dir}", flush=True)
    print(f"Config path: {args.config}", flush=True)
    print(
        f"Model: {'DeepAR' if args.deepar else ('TCN' if args.tcn else ('DecoderMLP' if args.decoder else 'LSTM'))}",
        flush=True)

    if args.gpu_check:
        print(f"GPU: {torch.cuda.is_available()}")
        sys.exit(0)

    with open(args.config, 'r', encoding='utf-8') as f:
        config_template = json.load(f)

    # Находим все папки с данными
    folders = find_data_folders(args.data_dir)
    if not folders:
        print(f"No data folders found in  {args.data_dir}", flush=True)
        sys.exit(1)
    print(f"\nStarting processing {len(folders)} folders", flush=True)

    for i, folder in enumerate(folders):
        print(f"\nProgress: {i + 1}/{len(folders)}", flush=True)
        print(args.tcn, flush=True)
        result = run_pipeline_for_folder(folder, config_template, args.decoder, args.single_dataset, tcn=args.tcn,
                                         deepar=args.deepar)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
