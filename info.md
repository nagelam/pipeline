project/
├── config/
│   └── config*.json
├── data/                    # Данные всавлять сюда
|   └── folder_with_data
|   |   └──223-ФЗ(папка)
|   |   └── ...
├── results/
|   |__ cl.sh очистка файлов с метриками               
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_model.py
|   |   └── *_model.py
│   ├── rolling_forecast/
│   │   ├── __init__.py
│   │   └── rolling_forecast*.py
│   └── pipeline.py
├── main.py
├── custom_period.py
└── info.md


main.py тут считыаются флаги и в зависимости от этого решается какую модель запустить, запускать ее для смердженных errors, request или отдельно
├── src/pipeline.py  Здесь запускается пайплайн для одного или 2 рядов сразу
Также тут создаются файлы с ошибками по дням 
    ├── src/data/data_loader.py  где считываются данные из сжатых файлов.
    ├──src/features/feature_engineering.py Здесь мы добавляем лаги и другие признаки
    └── src/rolling_forecast/rollingforecast* тут находится предсказание по окнам, оно вызывает файл модели
        └── src/models*_model тут находятся модели

После того как пайплайн выполнится надо вызвать
custom_period.py 
Который идет по дневным файлам с концом _daily, и в зависимости от того начало файла req или err агрегирует результаты по заданыым периодам
С ним есть недочеты изза его схемы работы. 
Проект поддерживает запуск нескольких моделей, и они последовательно выполнятся, получая например *lstm_16_daily*  *lstm_32_daily* но custom_period этого не различает, также он не различает название файлов когда лстм запускается для смердженных и отдельных рядов(при запуске отдельно рядов, файлы получают название дневных метрик _single_daily). 


Конфигурация:
Конфиги для моделей лежат в папке config. Конфиг для модели должен иметь такую структуру
 "model": {
    "requests": {
      "mlp_32_req": {
        "layers": [{
Можно заметить что такой конфиг поддерживает многослойную структуру
Важно чтобы модель лежала в requests или errors
Для моделей с 2 выходами конфигурацию модели надо класть в requests



Запуск
По умолчанию запускается лстм 
python main.py --data-dir data/
если имя кофига лстм не config_lstm
python main.py --data-dir data/ -c config/config_abc.json


Вызов остальных моделей 

python main.py --data-dir data/folder_with_data/ --deepar --config config/config_deepar.json
python main.py -d data/folder_with_data/ -c config/config_mlp.json --decoder
python main.py -d data/folder_with_data/ -c config/config_tcn.json --tcn

то есть в формате  
--data-dir data/
--[tcn, decoder, deepar]
–config config/[название конфига для модели]

опционально:
'--single-dataset', choices=['requests', 'errors'],
Эта команда прогонит модель только для 'requests' или только для 'errors'
python main.py --data-dir data/ --decoder --config config/config_model.json
 --single-dataset errors

Результаты модели лежат в result. Названия папок в result такиеже как и в папке data
Папки с результатами создаются автоматически

