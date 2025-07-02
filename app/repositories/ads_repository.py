import random

from app.helpers import read_json


class AdsRepository():
    def __init__(self, source_file_name):
        self.data = read_json(source_file_name)

    def filter_by_category(self, category_ids: list = None, shuffle: bool = True, items_num: int = 5) -> list:
        if category_ids is None:
            raise ValueError("You must provide category_ids")

        filtered_data = []
        for category_id in map(int, category_ids):
            items = [item for item in self.data if int(item['category_id']) == category_id]

            if shuffle:
                random.shuffle(items)

            filtered_data.extend(items[:items_num])

        if shuffle:
            random.shuffle(filtered_data)

        return filtered_data