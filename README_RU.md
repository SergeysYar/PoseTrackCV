# BrushPose AI

BrushPose AI — инженерный проект компьютерного зрения для детекции зубной щетки и оценки позы (центр + угол ориентации) на RGB-изображениях верхнего ракурса.

## Ключевые особенности
- Два пайплайна: классический CV и YOLOv8
- Единый CLI (`prepare-data`, `validate-data`, `train-yolo`, `infer`, `run-classical`, `evaluate`, `benchmark`, `generate-report`)
- Воспроизводимая конфигурация и модульная архитектура
- Двуязычная документация (EN/RU)
- Система бенчмаркинга и отчетности (CSV/JSON/Markdown/графики)

## Быстрый старт
```bash
python -m venv .venv
pip install -r requirements.txt
python src/cli.py validate-data --config configs/config.yaml
python src/cli.py benchmark --config configs/config.yaml
```

## Метрики
- IoU, detection accuracy, precision, recall
- Поддержка mAP@0.5 (placeholder)
- Ошибка центра, средняя/медианная ошибка угла
- FPS и время инференса
- Доля примеров с ошибкой угла `< 5°`

## Лицензия
MIT

