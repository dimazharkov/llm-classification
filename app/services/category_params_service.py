import random
from app.repositories.category_params_repository import CategoryParamsRepository
from app.helpers.prompt_helper import category_params_prompt
from app.ai.openai import OpenAI


class CategoryParamsService:
    def __init__(self, data, categories):
        self.data = data
        self.categories = categories
        self.max_items = 15

    def run(self, selected_categories: list = None):
        model = OpenAI(model_name="gpt-3.5-turbo", max_tokens=400)
        repository = CategoryParamsRepository()

        for category_id in selected_categories:
            params = self.process(category_id, model)
            params['title'] = self.categories[str(category_id)]["category_title"]
            print(f"{category_id}: {params}")
            repository.save_params(category_id, params)

        print("done!")

    def process(self, category_id, model):
        formatted_prompt = self.format_category_examples(category_id)

        model_result = model.predict(
            formatted_prompt
        )

        return model_result

    def format_category_examples(self, category_id: str) -> str:
        category_title = self.categories[str(category_id)]["category_title"]
        examples = []

        category_items = [item for item in self.data if int(item["category_id"]) == category_id]
        random.shuffle(category_items)

        for i, ad in enumerate(category_items[:self.max_items], start=1):
            examples.append(f"{i}. {ad['text']}")

        examples_text = "\n".join(examples)

        return category_params_prompt.format(category_title=category_title, examples=examples_text)