import pandas as pd
from pathlib import Path
import numpy as np
import argparse


def aggregate_by_periods(
        data_dict: dict,
        periods: list,
        metric_name: str = ""
) -> pd.DataFrame:
    """
    Агрегирует данные  по списку периодов.

    Args:
        data_dict: с индексом-датой и значениями метрики.
        periods: список словарей с ключами label, start, end
        metric_name: имя метрики для логирования (опционально).

    Returns:
        строки — периоды, столбцы — папки, значения — среднее за период
    """
    if not data_dict:
        print(f"No data available for {metric_name}")
        return pd.DataFrame()

    rows = []
    for period in periods:
        row = {"period": period["label"]}
        for folder_name, series in data_dict.items():
            # Фильтрация по [start, end] включительно
            mask = (series.index.date >= period["start"].date()) & \
                   (series.index.date <= period["end"].date())
            vals = series[mask]
            row[folder_name] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("period")
    return df


def create_custom_periods_aggregation(results_base_dir: str):
    """
    Создает по 4 отдельных датафрейма для каждой модели (LSTM, MLP, TCN, DeepAR) с агрегацией по настраиваемым периодам

    Args:
        results_base_dir: базовая директория с результатами

    Returns:
        словарь с датафреймами
    """
    results_path = Path(results_base_dir)

    if not results_path.exists():
        print(f"Directory {results_base_dir} not found!")
        return

    # Собираем все папки с результатами
    folder_paths = [p for p in results_path.iterdir() if p.is_dir()]

    # Словари для хранения данных по типам и моделям
    data_dicts = {
        'lstm': {
            'requests_mae': {},
            'requests_mse': {},
            'errors_mae': {},
            'errors_mse': {}
        },
        'mlp': {
            'requests_mae': {},
            'requests_mse': {},
            'errors_mae': {},
            'errors_mse': {}
        },
        'tcn': {
            'requests_mae': {},
            'requests_mse': {},
            'errors_mae': {},
            'errors_mse': {}
        },
        'deepar': {
            'requests_mae': {},
            'requests_mse': {},
            'errors_mae': {},
            'errors_mse': {}
        }
    }

    all_dates = set()

    print(f"Found {len(folder_paths)} result folders")

    # Обрабатываем каждую папку
    for folder_path in folder_paths:
        folder_name = folder_path.name
        print(f"Processing folder: {folder_name}")

        # Ищем daily файлы в папке
        daily_files = list(folder_path.glob("*_daily.csv"))

        if not daily_files:
            print(f"No daily files found in folder {folder_name}")
            continue

        for daily_file in daily_files:
            try:
                # Читаем данные
                df = pd.read_csv(daily_file, parse_dates=['day'])

                if df.empty:
                    continue

                # Проверяем наличие необходимых колонок
                if not all(col in df.columns for col in ['day', 'MAE_mean', 'MSE_mean']):
                    print(f" Missing required columns in file {daily_file.name}")
                    continue

                # Добавляем даты в общий набор
                all_dates.update(df['day'].dt.date)

                # Определяем тип модели по имени файла
                filename = daily_file.name.lower()
                if 'lstm' in filename:
                    model_type = 'lstm'
                elif 'mlp' in filename:
                    model_type = 'mlp'
                elif 'tcn' in filename:
                    model_type = 'tcn'
                elif 'deepar' in filename:
                    model_type = 'deepar'
                else:
                    print(f"Unknown model type in file: {daily_file.name}")
                    continue

                if filename.startswith('req'):
                    # Requests данные
                    data_dicts[model_type]['requests_mae'][folder_name] = df.set_index('day')['MAE_mean']
                    data_dicts[model_type]['requests_mse'][folder_name] = df.set_index('day')['MSE_mean']
                    print(f"  Loaded requests data ({model_type}): {len(df)} records")

                elif filename.startswith('err'):
                    # Errors данные
                    data_dicts[model_type]['errors_mae'][folder_name] = df.set_index('day')['MAE_mean']
                    data_dicts[model_type]['errors_mse'][folder_name] = df.set_index('day')['MSE_mean']
                    print(f"  Loaded errors data ({model_type}): {len(df)} records")
                else:
                    print(f"  Unknown file type: {daily_file.name}")

            except Exception as e:
                print(f"  Error processing {daily_file}: {e}")

    if not all_dates:
        print("No data found for processing")
        return

    # Определяем диапазон дат
    min_date = min(all_dates)
    max_date = max(all_dates)

    print(f"Date range: {min_date} - {max_date}")

    periods = [
        {
            'label': '10.07-10.20',
            'start': pd.to_datetime('2024-10-07', format='%Y-%m-%d'),
            'end': pd.to_datetime('2024-10-20', format='%Y-%m-%d')
        },
        {
            'label': '10.21-11.03',
            'start': pd.to_datetime('2024-10-21', format='%Y-%m-%d'),
            'end': pd.to_datetime('2024-11-03', format='%Y-%m-%d')
        },
        {
            'label': '11.04-11.17',
            'start': pd.to_datetime('2024-11-04', format='%Y-%m-%d'),
            'end': pd.to_datetime('2024-11-17', format='%Y-%m-%d')
        },
        {
            'label': '11.18-12.01',
            'start': pd.to_datetime('2024-11-18', format='%Y-%m-%d'),
            'end': pd.to_datetime('2024-12-01', format='%Y-%m-%d')
        },
        {
            'label': '12.02-12.15',
            'start': pd.to_datetime('2024-12-02', format='%Y-%m-%d'),
            'end': pd.to_datetime('2024-12-15', format='%Y-%m-%d')
        },
        {
            'label': '12.16-12.29',
            'start': pd.to_datetime('2024-12-16', format='%Y-%m-%d'),
            'end': pd.to_datetime('2024-12-29', format='%Y-%m-%d')
        },
        {
            'label': '12.30-01.01',
            'start': pd.to_datetime('2024-12-30', format='%Y-%m-%d'),
            'end': pd.to_datetime('2025-01-01', format='%Y-%m-%d')
        }
    ]

    print(f"Created {len(periods)} custom periods:")
    for p in periods:
        print(f"  - {p['label']}")

    print("\nCreating dataframes for LSTM, MLP, TCN and DeepAR")

    result_dfs = {}

    for model in ['lstm', 'mlp', 'tcn', 'deepar']:
        result_dfs[f'requests_mae_{model}'] = aggregate_by_periods(
            data_dicts[model]['requests_mae'], periods, f"Requests MAE {model.upper()}"
        )
        result_dfs[f'requests_mse_{model}'] = aggregate_by_periods(
            data_dicts[model]['requests_mse'], periods, f"Requests MSE {model.upper()}"
        )
        result_dfs[f'errors_mae_{model}'] = aggregate_by_periods(
            data_dicts[model]['errors_mae'], periods, f"Errors MAE {model.upper()}"
        )
        result_dfs[f'errors_mse_{model}'] = aggregate_by_periods(
            data_dicts[model]['errors_mse'], periods, f"Errors MSE {model.upper()}"
        )

    # Сохраняем результаты
    output_dir = Path(results_base_dir)

    files_created = []

    for key, df in result_dfs.items():
        if not df.empty:
            metric, model = key.rsplit('_', 1)
            file_name = f"custom_periods_{metric}_{model}.csv"
            file_path = output_dir / file_name
            df.to_csv(file_path)
            files_created.append(str(file_path))

    print(f"\nSaved files:")
    for file_path in files_created:
        print(f"- {file_path}")

    # Выводим примеры данных для каждого датафрейма
    for key, df in result_dfs.items():
        if not df.empty:
            metric, model = key.rsplit('_', 1)
            print(f"\nSample data ({metric.capitalize()} {model.upper()} dataframe):")
            print(df.head())

    return result_dfs


def main():
    parser = argparse.ArgumentParser(
        description='Создание сводных таблиц по настраиваемым периодам для LSTM, MLP, TCN и DeepAR'
    )
    parser.add_argument(
        '--results-dir', '-r',
        default='results',
        help='Папка с результатами'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CREATING SUMMARY TABLES FOR CUSTOM PERIODS LSTM, MLP, TCN and DEEPAR")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")

    result = create_custom_periods_aggregation(args.results_dir)

    if result:
        print("\nProcessing completed successfully. Created (4 dataframes for each model).")
    else:
        print("\nError occurred during data processing.")


if __name__ == "__main__":
    main()
