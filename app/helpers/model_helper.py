import json
from pathlib import Path

from app.helpers.file_helper import read_json, write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
prompt_file_path = PROJECT_ROOT / "static" / "comparison_prompts.json"

def save_comparison_prompt_to_file(prompt: str, key: str):
    data = read_prompt_from_file()
    if data is None:
        data = {}

    data[key] = prompt
    write_json(prompt_file_path, data)

    return data

def save_interim_results(data, file_name = "interim_results.json"):
    file_path = PROJECT_ROOT / "static" / file_name
    write_json(file_path, data)


def read_prompt_from_file() -> dict:
    prompts = read_json(prompt_file_path)
    return prompts

def get_comparison_prompt_from_file(key) -> str | None:
    try:
        prompts = read_json(prompt_file_path)

        if not isinstance(prompts, dict):
            raise ValueError("Файл не содержит словарь с промптами.")

        if key not in prompts:
            raise KeyError(f"Ключ '{key}' не найден в файле промптов.")

        return prompts[key]

    except FileNotFoundError:
        print(f"Файл не найден: {prompt_file_path}")
    except json.JSONDecodeError:
        print(f"Файл {prompt_file_path} содержит некорректный JSON.")
    except KeyError as e:
        print(str(e))
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")

    return None