import itertools
import random

from app.repositories.category_diff_repository import CategoryDiffRepository
from app.helpers.prompt_helper import category_diff_prompt
from app.ai.openai import OpenAI


class DefferService:
    def __init__(self, data, categories):
        self.data = data
        self.categories = categories
        self.max_items = 10
        self.category_items = {}

    def run(self, selected_categories: list = None):
        model = OpenAI(model_name="gpt-3.5-turbo", max_tokens=200)
        repository = CategoryDiffRepository()

        for cat1, cat2 in itertools.combinations(selected_categories, 2):
            difference = self.process(cat1, cat2, model)
            print(f"{cat1} -> {cat2}: {difference}")
            repository.save_difference(cat1, cat2, difference)

        print("done!")

    def process(self, cat1, cat2, model):
        category1 = self.format_category_examples(cat1)
        category2 = self.format_category_examples(cat2)

        formatted_prompt = category_diff_prompt.format(
            category1=category1, category2=category2
        )

        model_result = model.predict(
            formatted_prompt
        )

        return model_result

    def format_category_examples(self, category_id: str) -> str:
        category_title = self.categories[str(category_id)]["category_title"]
        output = [f"{category_title}, примеры объявлений:"]

        if category_id not in self.category_items:
            category_items = [item for item in self.data if int(item["category_id"]) == category_id]
            random.shuffle(category_items)
            self.category_items[category_id] = category_items[:self.max_items]

        category_items = self.category_items[category_id]

        for i, ad in enumerate(category_items, start=1):
            output.append(f"{i}. {ad['text']}")

        return "\n".join(output)
