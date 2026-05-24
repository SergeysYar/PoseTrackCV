# BrushPose AI
### Детекция зубной щетки и оценка 2D-позы в задачах tabletop computer vision

Инженерная система компьютерного зрения для локализации объекта и оценки его ориентации на изображениях верхнего ракурса.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#установка)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](#функциональные-возможности)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](#обучение)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#лицензия)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen.svg)](#направления-развития)

> Практико-ориентированный CV-конвейер для задач пространственной локализации и инженерной оценки позы.

---

## Обзор проекта
**BrushPose AI** предназначен для обнаружения зубной щетки на однотонной столешнице и последующей оценки:
- координат центра объекта \((x_{center}, y_{center})\)
- угла ориентации \(\theta \in [0^\circ, 180^\circ)\)

Архитектура объединяет:
- классический CV-подход (интерпретируемость и геометрическая прозрачность)
- детекцию на основе YOLOv8 (масштабируемость и устойчивость)
- единую систему оценки качества, бенчмаркинга и отчетности

Репозиторий оформлен в формате production-quality портфолио для ML/CV-инженерии и исследовательских демонстраций.

---

## Постановка задачи
**Входные данные**:
- RGB-изображения или видеокадры верхнего ракурса с рабочей поверхностью.

**Требуемые выходы**:
- ограничивающий прямоугольник (bounding box)
- координаты центра объекта в пиксельной системе
- угол ориентации относительно оси \(x\)
- визуализация предсказаний и метрики качества

**Основные сложности**:
- вариативность освещения и теней
- малая цветовая контрастность объекта и фона
- неустойчивость сегментации для вытянутых тонких объектов
- геометрическая неоднозначность ориентации
- погрешности в угловой разметке

---

## Практические сценарии применения
- **Робототехника**: предварительная оценка позы перед захватом
- **Промышленная автоматизация**: alignment в pick-and-place системах
- **Умное производство**: контроль размещения деталей на рабочей зоне
- **Складская автоматизация**: ориентационно-зависимая сортировка
- **Визуальный контроль качества**: проверка корректности укладки объектов
- **Tabletop localization**: координатная привязка объектов в ограниченной сцене

---

## Функциональные возможности
- Детекция зубной щетки на RGB-кадрах
- Оценка координат центра объекта
- Оценка угла ориентации методами `minAreaRect` и PCA
- Классический пайплайн: HSV-сегментация, морфология, контуры
- YOLOv8 пайплайн: обучение и инференс с конфигурацией параметров
- Визуализация: bbox, центр, вектор ориентации, значение угла
- Поддержка одиночного и пакетного инференса
- Подсистема метрик и автоматической генерации отчетов
- Сравнительный бенчмаркинг нескольких подходов
- Инструменты подготовки и валидации датасета
- Единый CLI-интерфейс (`python src/cli.py ...`)
- Двуязычная инженерная документация

---

## Архитектура проекта
Система построена модульно, с явным разделением ответственности.

```text
Сбор/валидация данных
          |
          v
     Слой детекции
 (Classical CV / YOLOv8)
          |
          v
  Слой оценки 2D-позы
 (центр + ориентация)
          |
          v
 Метрики и бенчмаркинг
          |
          v
Визуализация и отчеты
```

**Назначение модулей**

| Модуль | Назначение |
|---|---|
| `src/data` | извлечение кадров, разбиение train/val/test, валидация, конвертация аннотаций |
| `src/detection` | обучение и инференс YOLO, интерфейсы детекции |
| `src/pose` | геометрические функции, классическая оценка позы, PCA-ориентация |
| `src/evaluation` | метрики, бенчмарк, сравнение методов, генерация отчетов |
| `src/visualization` | оверлеи предсказаний, графики, сравнительные артефакты |
| `src/cli.py` | единая точка запуска сценариев |

Плейсхолдер схемы: `assets/pipeline.png`

---

## Структура датасета
```text
data/
├── raw/
├── images/
├── annotations/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Принципы организации**
- Имена изображений: `scene_<id>.jpg` или `frame_<id>.jpg`
- Для каждого изображения отдельный label-файл с тем же stem
- Опционально добавляется поле угла ориентации

**Система координат**
- начало координат: левый верхний угол
- ось \(x\): вправо
- ось \(y\): вниз

---

## Сбор данных
Рекомендуемый протокол:
- фиксированная камера над рабочей поверхностью
- преимущественно однородный матовый фон
- контролируемое рассеянное освещение
- мультисценарный сбор:
  - различные покрытия стола
  - вариации экспозиции/баланса белого
  - широкий диапазон углов и положений объекта

Для устойчивости модели:
- включать сложные случаи (блики, тени, слабый контраст)
- избегать переобучения на однотипных шаблонных сценах
- выравнивать распределение углов для снижения смещения

---

## Формат аннотаций
### YOLO-формат
```text
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm> [angle_deg]
```

Пример:
```text
0 0.532812 0.471354 0.285937 0.092708 37.5
```

### CSV-формат
```csv
image,x1,y1,x2,y2,class_id,angle_deg
frame_000123.jpg,412,265,701,332,0,37.5
```

### Угловые соглашения
- диапазон: \([0^\circ, 180^\circ)\)
- осевая ориентация без различения направлений «голова-хвост»

---

## Установка
### 1. Клонирование
```bash
git clone https://github.com/<your-org>/BrushPoseAI.git
cd BrushPoseAI
```

### 2. Виртуальное окружение
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Зависимости
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Быстрый старт
```bash
python src/cli.py validate-data --config configs/config.yaml
python src/cli.py run-classical --image data/images/sample.jpg --output outputs/images/classical_sample.png
python src/cli.py infer --method yolo --image data/images/sample.jpg --weights yolov8n.pt --output outputs/images/yolo_sample.png
python src/cli.py benchmark --config configs/config.yaml
```

Плейсхолдеры демонстраций:
- `assets/demo.gif`
- `assets/results_example.png`

---

## Обучение
### Последовательность подготовки
1. Сбор/извлечение изображений
2. Разметка bbox и, при необходимости, угла
3. Конвертация аннотаций в YOLO-формат
4. Разбиение train/val/test
5. Валидация целостности и качества разметки

### Запуск обучения YOLO
```bash
python src/cli.py train-yolo --config configs/yolo_config.yaml
```

### Основные гиперпараметры

| Параметр | Смысл |
|---|---|
| `epochs` | число эпох оптимизации |
| `imgsz` | входное разрешение изображения |
| `batch` | размер мини-батча |
| `conf` | порог confidence для инференса |
| `device` | вычислительное устройство (CPU/GPU) |

Чекпоинты и логи сохраняются в выходные директории, заданные конфигурацией.

---

## Инференс
### Одиночное изображение
```bash
python src/cli.py infer --method classical --image data/images/sample.jpg --output outputs/images/pred_classical.png
python src/cli.py infer --method yolo --image data/images/sample.jpg --weights yolov8n.pt --output outputs/images/pred_yolo.png
```

### Пакетная обработка (через оценку)
```bash
python src/cli.py evaluate --method classical --image-dir data/test/images --label-dir data/test/labels
python src/cli.py evaluate --method yolo --image-dir data/test/images --label-dir data/test/labels --weights yolov8n.pt
```

Результаты включают визуализации, таблицы метрик и сводные отчеты.

---

## Оценка качества
Оценка выполняется на отложенной тестовой выборке:
1. запуск детекции и оценки позы
2. сопоставление предсказаний с ground truth
3. вычисление детекционных и геометрических метрик
4. агрегация статистик по методам
5. сохранение CSV/JSON/Markdown отчетов

Сравнение методов:
```bash
python src/cli.py benchmark --config configs/config.yaml
```

---

## Метрики
### IoU (Intersection-over-Union)
\[
\mathrm{IoU} = \frac{|B_{pred}\cap B_{gt}|}{|B_{pred}\cup B_{gt}|}
\]

### Ошибка центра
\[
e_c = \sqrt{(x_{pred}-x_{gt})^2 + (y_{pred}-y_{gt})^2}
\]

### Ошибка угла (с периодичностью 180°)
\[
e_\theta = \min\left(|\theta_{pred}-\theta_{gt}|,\ 180^\circ-|\theta_{pred}-\theta_{gt}|\right)
\]

### Дополнительные показатели
- Precision / Recall
- mAP@0.5 (поддержка placeholder)
- медианная угловая ошибка
- доля выборки с \(e_\theta < 5^\circ\)
- latency (мс/кадр)
- FPS

Инженерная интерпретация:
- IoU отражает качество локализации объекта
- ошибки центра/угла определяют пригодность для задач захвата
- latency и FPS определяют применимость в реальном времени

---

## Примеры выходных артефактов
Ожидаемые результаты:
- оверлеи предсказаний: bbox + центр + стрелка + значение угла
- гистограммы угловых ошибок
- сравнительные бенчмарк-графики
- машиночитаемые summary-файлы

Плейсхолдеры:
- `assets/results_example.png`
- `assets/benchmark_plot.png`
- `assets/architecture.png`

---

## Ограничения
- чувствительность классической сегментации к сильным изменениям освещения
- снижение качества на неоднородных/зашумленных фонах
- неоднозначность ориентации для визуально симметричных случаев
- деградация при частичной окклюзии объекта
- зависимость обобщающей способности от вариативности датасета

---

## Направления развития
- интеграция instance segmentation для уточнения оси объекта
- переход к transformer-based детекторам
- keypoint-подход для снятия осевой неоднозначности
- добавление глубины (RGB-D) для 3D-расширения
- оптимизация реального времени (TensorRT/ONNX/OpenVINO)
- интеграция в ROS2-контуры
- edge-deployment профили (Jetson/industrial edge)

---

## Лицензия
Проект распространяется по лицензии **MIT**.  
Подробности: [LICENSE](LICENSE).

Примечания по атрибуции:
- сохранение лицензии и копирайт-нотиса в производных работах
- соблюдение лицензий сторонних библиотек

---

## Благодарности
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- сообщество open-source computer vision

---

## Примеры CLI-команд
```bash
# Данные
python src/cli.py prepare-data --config configs/config.yaml
python src/cli.py validate-data --config configs/config.yaml

# Обучение
python src/cli.py train-yolo --config configs/yolo_config.yaml

# Инференс
python src/cli.py run-classical --image data/images/sample.jpg --output outputs/images/classical.png
python src/cli.py infer --method yolo --image data/images/sample.jpg --weights yolov8n.pt --output outputs/images/yolo.png

# Оценка
python src/cli.py evaluate --method classical --image-dir data/test/images --label-dir data/test/labels
python src/cli.py evaluate --method yolo --image-dir data/test/images --label-dir data/test/labels --weights yolov8n.pt

# Бенчмарк и отчеты
python src/cli.py benchmark --config configs/config.yaml
python src/cli.py generate-report
```

