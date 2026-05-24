# Подготовка датасета для BrushPose AI

## Назначение
Подсистема подготовки данных обеспечивает воспроизводимое формирование набора изображений для задач:
- локализации зубной щетки на столешнице;
- оценки координат центра объекта;
- оценки угла ориентации в диапазоне `[0, 180]`.

Каждая запись аннотации должна содержать:
- `x_min, y_min, x_max, y_max` (ограничивающий прямоугольник);
- `x_center, y_center` (центр объекта);
- `angle_deg` (угол ориентации).

## Рекомендации по съемке
- Камера фиксируется над рабочей поверхностью (вид сверху).
- Оптическая ось ориентирована близко к нормали поверхности.
- Фон предпочтительно однородный, матовый.
- Освещение рассеянное и стабильное, без жестких теней.
- В датасет включаются разнообразные позиции и ориентации объекта.

## Структура каталогов
```text
data/
  images/
    image_000001.jpg
    image_000002.jpg
  annotations/
    annotations.csv
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

## Формат аннотаций
Основной файл: `data/annotations/annotations.csv`

Обязательные столбцы:
- `filename`
- `width`
- `height`
- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `x_center`
- `y_center`
- `angle_deg`
- `class_name`

Класс по умолчанию: `toothbrush`.

Система координат:
- начало координат: левый верхний угол;
- ось `x`: вправо;
- ось `y`: вниз.

Угловая конвенция:
- осевая ориентация объекта;
- допустимый диапазон: `[0, 180]`.

## 1) Создание шаблона датасета
Скрипт нормализует имена изображений (`image_000001.jpg`, ...) и формирует шаблон CSV:

```bash
python src/data/collect_dataset_template.py \
  --input-dir data/raw \
  --output-dir data/images \
  --annotations-out data/annotations/annotations.csv \
  --copy \
  --convert-to-jpg \
  --class-name toothbrush
```

Особенности:
- поддерживаемые расширения: `.jpg .jpeg .png .bmp .webp`;
- опциональное изменение размера: `--resize-width --resize-height`;
- режим перемещения исходников: `--move` (вместо `--copy`).

## 2) Валидация датасета
```bash
python src/data/validate_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --report-out outputs/reports/dataset_validation.md
```

Строгий режим:
```bash
python src/data/validate_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --report-out outputs/reports/dataset_validation.md \
  --strict
```

Проверяемые условия:
- непустой `filename`;
- существование файла изображения и его читаемость;
- согласованность `width/height` с реальным размером;
- корректность границ bbox и условие `x_min < x_max`, `y_min < y_max`;
- принадлежность центра ограничивающему прямоугольнику;
- `angle_deg` в диапазоне `[0, 180]`;
- непустой `class_name`;
- отсутствие дубликатов файлов.

В нестрогом режиме допускается отсутствие `width/height` и `x_center/y_center` с последующим восстановлением.

## 3) Разбиение train/val/test и экспорт меток
```bash
python src/data/split_dataset.py \
  --images-dir data/images \
  --annotations data/annotations/annotations.csv \
  --output-dir data \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --copy-images \
  --format both
```

Опция `--format`:
- `csv`: `annotations.csv` в каждом split;
- `yolo`: YOLO `.txt` + `angle_labels.csv`;
- `both`: оба формата.

Автоматически формируются статистики:
- `outputs/metrics/dataset_stats.csv`
- `outputs/reports/dataset_stats.md`

## 4) Конвертация аннотаций
Из BrushPose CSV в YOLO:
```bash
python src/data/convert_annotations.py \
  --input data/annotations/annotations.csv \
  --images-dir data/images \
  --output-dir data/yolo_labels \
  --from-format brushpose-csv \
  --to-format yolo \
  --class-id 0
```

Из BrushPose CSV в pose CSV:
```bash
python src/data/convert_annotations.py \
  --input data/annotations/annotations.csv \
  --images-dir data/images \
  --output-dir data/pose_labels \
  --from-format brushpose-csv \
  --to-format pose-csv
```

## Типичные ошибки
- несогласованность размеров изображения и полей `width/height`;
- bbox выходит за границы изображения;
- ошибочный диапазон углов (`[-180,180]` или `[0,360]`);
- дублирующиеся имена файлов в `annotations.csv`;
- смешение расширений и регистра в `filename`.

## Контрольный список качества
- [ ] все изображения читаются OpenCV;
- [ ] все `filename` из CSV присутствуют в `data/images`;
- [ ] bbox корректен и внутри границ кадра;
- [ ] центр согласован с bbox;
- [ ] углы находятся в диапазоне `[0, 180]`;
- [ ] классы заполнены и консистентны;
- [ ] разбиение воспроизводимо при фиксированном `seed`;
- [ ] отчеты валидации и статистики успешно сформированы.

