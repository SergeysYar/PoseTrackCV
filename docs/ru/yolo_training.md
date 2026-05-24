# YOLO-подсистема (BrushPose AI)

## Назначение
YOLO-модуль обеспечивает инженерный контур детекции зубной щетки:
- экспорт аннотаций BrushPose в формат YOLO;
- обучение модели Ultralytics YOLO;
- опциональную валидацию после обучения;
- инференс по изображению/каталогу (дополнительно поддерживается видео);
- сохранение визуализаций и CSV-предсказаний.

## Входной формат аннотаций
Исходный CSV (`BrushPose format`):
- `filename,width,height,x_min,y_min,x_max,y_max,x_center,y_center,angle_deg,class_name`

Для стандартной YOLO-детекции используются только bbox и class label.  
Поле `angle_deg` сохраняется отдельно в split-CSV и не участвует в обучении детектора.

## Экспорт в YOLO-формат
```bash
python src/detection/export_yolo_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --output-dir data/yolo_dataset \
  --seed 42 \
  --copy-images
```

Формируемая структура:
```text
data/yolo_dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
  angle_labels/{train_angles.csv,val_angles.csv,test_angles.csv}
  dataset.yaml
  export_report.md
```

## Файл dataset.yaml
```yaml
path: data/yolo_dataset
train: images/train
val: images/val
test: images/test
names:
  0: toothbrush
```

## Обучение
```bash
python src/detection/train_yolo.py \
  --data data/yolo_dataset/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --validate
```

Результаты:
- директория эксперимента Ultralytics (`project/name`);
- веса `weights/best.pt`, `weights/last.pt`;
- отчет `training_summary.md`.

## Валидация
Валидация запускается флагом `--validate` в скрипте обучения.  
При наличии метрик они записываются в `training_summary.md`.

## Инференс
```bash
python src/detection/infer_yolo.py \
  --weights runs/brushpose_yolo/train/weights/best.pt \
  --input data/test/images \
  --output-dir outputs/images/yolo \
  --csv-out outputs/metrics/yolo_predictions.csv \
  --conf 0.25
```

Дополнительные опции:
- `--save-txt`: сохранить нормализованные YOLO-предсказания;
- `--save-crops`: сохранить вырезки детекций;
- `--config`: задать параметры через YAML-конфиг.

## Формат CSV предсказаний
Поля:
- `filename,status,class_id,class_name,confidence`
- `x1,y1,x2,y2,center_x,center_y,width,height`
- `image_width,image_height,processing_time_ms,message`

Статусы:
- `detected`
- `no_detection`
- `failed`

## Рекомендованные базовые гиперпараметры
- модель: `yolov8n.pt`
- `imgsz`: `640`
- `batch`: `8` (увеличивать при достаточной VRAM)
- `epochs`: `50`
- `conf` (инференс): `0.25`
- `iou` (NMS): `0.5`

## Типовые проблемы
1. `Ultralytics package is not installed`
2. `dataset.yaml not found`
3. `weights file not found`
4. `No supported images found for inference`
5. Пустой экспорт из-за несовпадения класса (`class_name != toothbrush`)

## Устранение ошибок
- Установить пакет: `pip install ultralytics`.
- Проверить корректность путей в `dataset.yaml` относительно `path`.
- Проверить числовые поля аннотаций (`width,height,x_min...`).
- Убедиться, что все `filename` из CSV существуют в `--images-dir`.
- Начинать с `yolov8n.pt`, затем масштабировать модель при необходимости.

