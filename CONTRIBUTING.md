# Contributing to BrushPose AI

Thank you for contributing to BrushPose AI. This repository is designed as a production-style computer vision engineering project focused on reproducibility, clarity, and measurable quality.

## 1. Project Scope

BrushPose AI contains:
- dataset preparation tools
- classical OpenCV pose estimation
- YOLO training and inference utilities
- evaluation and reporting modules
- multi-method benchmarking
- visualization tooling

Before contributing, read:
- `README.md`
- `docs/en/architecture.md`
- `docs/en/evaluation.md`

## 2. Repository Structure

Key directories:
- `src/data`: dataset preparation and validation
- `src/pose`: classical CV and geometry logic
- `src/detection`: YOLO export/train/infer workflows
- `src/evaluation`: metrics, evaluation, report generation
- `src/visualization`: overlays and plots
- `scripts`: orchestration scripts for full runs
- `docs`: bilingual technical documentation
- `tests`: unit tests

## 3. Development Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## 4. Development Workflow

1. Create a feature branch from `main`.
2. Implement a focused change set.
3. Run lint/tests locally.
4. Update docs/config examples when behavior changes.
5. Open a pull request with evidence (metrics/logs/screenshots).

## 5. Code Style and Quality Rules

- Python 3.10+
- Use `pathlib.Path` for paths
- Add type hints to public functions
- Add concise docstrings for modules/classes/functions
- Avoid hardcoded absolute paths
- Keep CLI messages explicit and actionable
- Do not duplicate metric formulas across modules

Import order:
1. standard library
2. third-party
3. local package modules

## 6. Commit Message Convention

Use short, descriptive, imperative messages.

Examples:
- `feat(evaluation): add angle symmetry handling`
- `fix(cli): align benchmark flags with script`
- `docs(readme): update unified cli examples`
- `chore(config): refresh production defaults`

## 7. Pull Request Guidelines

Each PR should include:
- concise problem statement
- implementation summary
- impacted modules/files
- backward compatibility notes
- test evidence

Recommended PR checklist:
- [ ] No hardcoded local paths
- [ ] CLI examples verified
- [ ] Documentation updated
- [ ] Tests pass locally

## 8. Testing Recommendations

Run full test suite:

```bash
pytest
```

Run targeted tests:

```bash
pytest tests/test_metrics.py
pytest tests/test_geometry.py
pytest tests/test_dataset.py
```

For pipeline changes, include at least one dry-run command in PR description.

## 9. Reporting Issues

Use GitHub Issues and include:
- environment (OS, Python, package versions)
- command executed
- expected behavior
- actual behavior
- full error log snippet

Bug reports without reproducible steps may be delayed.

## 10. Branch Naming

Suggested patterns:
- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `chore/<short-description>`

Examples:
- `feature/yolo-export-validation`
- `fix/classical-angle-normalization`
- `docs/readme-cli-polish`

---

By contributing, you agree that your contributions are provided under the project license.
