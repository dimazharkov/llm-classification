import csv
import json
import os
from pathlib import Path
from typing import Any

from app.helpers.text_helper import plain_text


def csv_to_json(csv_path: str, json_path: str) -> list:
    csv_file_path = Path(csv_path)

    # fieldnames = ["category_title", "category_id", "title", "text"]
    fieldnames = [
        "category_parent_id",
        "category_parent_title",
        "category_id",
        "category_title",
        "advert_title",
        "advert_text",
    ]

    with csv_file_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        data = list(reader)

    for item in data:
        item["parent_title"] = plain_text(item["category_parent_title"])
        item["parent_id"] = int(item["category_parent_id"]) if item["category_parent_id"].isdigit() else 0
        item["category_title"] = plain_text(item["category_title"])
        item["category_id"] = int(item["category_id"]) if item["category_id"].isdigit() else 0
        item["title"] = plain_text(item["advert_title"])
        item["text"] = plain_text(item["advert_text"])

    needed_keys = [
        "parent_title",
        "parent_id",
        "category_title",
        "category_id",
        "title",
        "text",
    ]

    clean_data = [{key: item[key] for key in needed_keys if key in item} for item in data]

    clean_data[:] = [item for item in clean_data if 50 < len(item.get("text", "")) <= 600]

    write_json(json_path, clean_data)

    return clean_data


def read_json(json_path: str | Path) -> Any | None:
    json_file_path = Path(json_path)
    if not json_file_path.exists():
        return None
    with json_file_path.open("r", encoding="utf-8") as jsonfile:
        return json.load(jsonfile)


def write_json(json_path: str | Path, data: dict | list) -> None:
    json_file_path = Path(json_path)
    with json_file_path.open("w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)
        jsonfile.flush()
        os.fsync(jsonfile.fileno())

    print(f"Файл успешно сохранен в {json_file_path}")


def load_prompt(prompt_path) -> str:
    PROMPT_FILE = Path(prompt_path)
    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        return f.read()
