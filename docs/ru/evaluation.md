# Подсистема оценки качества

## Назначение
Модуль оценки сопоставляет предсказания с эталонной разметкой и отвечает на ключевые вопросы:
- насколько корректно обнаружен объект;
- насколько точно предсказан ограничивающий прямоугольник;
- какова ошибка координат центра;
- какова точность оценки угла ориентации (при наличии угловых предсказаний);
- каковы вычислительные характеристики (время обработки, FPS);
- какие типы ошибок встречаются чаще всего.

Подсистема пригодна для оценки classical CV, YOLO и будущих комбинированных методов.

## Форматы входных CSV
Эталонная разметка (ground truth):
- `filename,width,height,x_min,y_min,x_max,y_max,x_center,y_center,angle_deg,class_name`

Предсказания (стандартный формат):
- `filename,status,x_min,y_min,x_max,y_max,x_center,y_center,angle_deg,confidence,processing_time_ms,message`

Поддерживаются также YOLO-варианты имен столбцов:
- `x1,y1,x2,y2,center_x,center_y`

Внутренне они автоматически приводятся к единому формату.

## Логика сопоставления
- Сопоставление выполняется **по имени файла**, без учета префиксов пути.
- Если для одного файла несколько предсказаний:
  1. приоритет у успешных (`success`/`detected`);
  2. при наличии confidence выбирается максимальный confidence;
  3. иначе выбирается первое успешное предсказание.
- Отсутствие предсказания трактуется как `missed_detection`.
- Статусы `failed` и `no_detection` считаются неуспешной детекцией.

## Поддерживаемые метрики
- `IoU` — качество пересечения bbox.
- `center_error_px` — евклидова ошибка центра (в пикселях).
- `angle_error_deg` — минимальная угловая ошибка с учетом 180° симметрии.
- `detection_accuracy`
- `precision`, `recall`, `f1`
- `map_50_proxy` — доля выборок с `IoU >= 0.5` (упрощенный proxy, не COCO mAP).
- Метрики производительности: mean/median processing time, FPS.

## Обработка отсутствия углов
Если в предсказаниях отсутствует `angle_deg`:
- угловые метрики не вычисляются;
- в summary отмечаются как `unavailable`;
- соответствующие строки получают `error_type = no_angle_prediction`.

## Анализ ошибок
Автоматически формируются типы ошибок:
- `ok`
- `missed_detection`
- `low_iou`
- `high_center_error`
- `high_angle_error`
- `invalid_prediction`
- `missing_ground_truth`
- `no_angle_prediction`

Отчёт включает:
- распределение по типам ошибок;
- топ-10 худших примеров по IoU;
- топ-10 худших примеров по угловой ошибке.

## Пример запуска
```bash
python src/evaluation/evaluate_predictions.py \
  --ground-truth data/test/labels/annotations.csv \
  --predictions outputs/metrics/classical_cv_predictions.csv \
  --output-dir outputs/reports/classical_cv_eval \
  --method-name classical_cv_pca \
  --iou-threshold 0.5 \
  --angle-threshold 5 \
  --report-format both
```

## Выходные файлы
- `metrics.csv` — построчные метрики по каждому файлу;
- `summary_metrics.json` — агрегированные метрики и счетчики;
- `benchmark_report_en.md` — англоязычный markdown-отчёт;
- `benchmark_report_ru.md` — русскоязычный markdown-отчёт.

## Интерпретация результатов
- Рост `mean_iou`, `precision`, `recall` указывает на улучшение детекции.
- Снижение `mean_center_error_px` означает более точную геометрическую локализацию.
- Снижение `mean_angle_error_deg` означает более корректную оценку ориентации.
- Рост `fps` повышает применимость в задачах реального времени.

