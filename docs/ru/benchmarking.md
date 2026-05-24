# Бенчмаркинг (сравнение нескольких методов)

## Назначение
Скрипт `run_benchmark.py` выполняет воспроизводимое сравнение трёх подходов:
- `classical_min_area_rect`
- `classical_pca`
- `yolo_geometric`

Оркестрация опирается на уже реализованные модули инференса и оценки качества, после чего агрегирует результаты в единые таблицы и отчёты.

## Необходимые входы
- `--images-dir`: каталог тестовых изображений
- `--ground-truth`: CSV с эталонной разметкой
- `--output-dir`: базовый каталог для артефактов
- `--yolo-weights`: путь к весам YOLO (опционально при пропуске YOLO)

## Пример команды
```bash
python scripts/run_benchmark.py \
  --images-dir data/test/images \
  --ground-truth data/test/labels/annotations.csv \
  --output-dir outputs \
  --yolo-weights runs/brushpose_yolo/train/weights/best.pt \
  --methods classical_min_area_rect classical_pca yolo_geometric \
  --iou-threshold 0.5 \
  --angle-threshold 5 \
  --language both \
  --skip-yolo-if-missing
```

## Структура выходов
```text
outputs/
  images/benchmark/<method>/
  metrics/benchmark/
    <method>_predictions.csv
    <method>_metrics.csv
    <method>_summary.json
    benchmark_results.csv
    benchmark_summary.json
  metrics/benchmark_results.csv
  reports/
    benchmark_en.md
    benchmark_ru.md
    benchmark_logs/
```

## Интерпретация `benchmark_results.csv`
Ключевые поля:
- `detection_accuracy`
- `mean_iou`, `median_iou`
- `mean_center_error_px`
- `mean_angle_error_deg`
- `mean_processing_time_ms`, `fps`
- `status`, `notes`

Поля `status` и `notes` используются для фиксации частичных отказов, пропусков методов и недоступных метрик.

## Угловые метрики для YOLO
Стандартная YOLO-детекция предсказывает bbox и класс, но не угол ориентации.  
Если `yolo_geometric` не возвращает `angle_deg`, угловые метрики остаются пустыми, а причина фиксируется в `notes`.

## Типовые сценарии ошибок
- отсутствует каталог изображений или файл ground truth;
- отсутствуют веса YOLO;
- недоступен пакет Ultralytics;
- для метода сформирован пустой CSV предсказаний;
- сбой оценки для отдельного метода.

При этом бенчмарк не прерывается полностью: проблемный метод помечается как `failed`/`skipped`, остальные продолжают выполняться.

## Примечания по воспроизводимости
- По возможности фиксируйте seed в компонентах инференса.
- Используйте неизменяемый тестовый набор и стабильный CSV разметки.
- Анализируйте логи методов в `outputs/reports/benchmark_logs/`.
- Определения метрик приведены в `docs/ru/evaluation.md`.

