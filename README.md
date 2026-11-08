# Симулятор регулятора (PyQt6)

Графический симулятор системы управления температурой с PID-регулятором, адаптивным PID и Q-learning агентом. Интерфейс построен на PyQt6, графики — matplotlib.

## Структура проекта

| Файл | Назначение |
|---|---|
| [`simulyator_qt.py`](simulyator_qt.py) | Главный файл — GUI-приложение (запускать его) |
| [`praktika_kontury.py`](praktika_kontury.py) | Модель объекта (`Plant`), PID, адаптивный PID, Q-learning агент |

## Требования

- Python **3.10+** (разработано на Python 3.10–3.12)
- Windows / Linux / macOS

## Зависимости

- **PyQt6** — графический интерфейс
- **numpy** — вычисления
- **matplotlib** — графики (бэкенд QtAgg)

Полный список — в [`requirements.txt`](requirements.txt).

## Установка и запуск

1. (Рекомендуется) создать виртуальное окружение:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Установить зависимости:

```powershell
pip install -r requirements.txt
```

3. Запустить симулятор:

```powershell
python simulyator_qt.py
```

Быстрый вариант без виртуального окружения:

```powershell
pip install PyQt6 numpy matplotlib
python simulyator_qt.py
```

## Готовый EXE

Собранная версия для Windows: **`dist\Simulyator.exe`** — запускается двойным кликом, установка Python не требуется.

> Примечание: при первом запуске exe может открываться 5–10 секунд (распаковка) — это нормально.

## Устранение неполадок

| Проблема | Решение |
|---|---|
| `ModuleNotFoundError: No module named 'PyQt6'` | Установить зависимости: `pip install -r requirements.txt` |
| Приложение не запускается в venv из VS Code | Выбрать интерпретатор venv: `Ctrl+Shift+P` → *Python: Select Interpreter* |
| Ошибка кодировки в консоли | В коде уже есть `sys.stdout.reconfigure(encoding="utf-8")`; при необходимости выполнить `chcp 65001` |
| Антивирус ругается на exe | PyInstaller-файлы иногда дают ложное срабатывание — добавить `Simulyator.exe` в исключения антивируса |
