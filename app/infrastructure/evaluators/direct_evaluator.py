from app.evaluators.base import BaseEvaluator
from app.helpers.prompt_helper import classification_prompt


class DirectEvaluator(BaseEvaluator):
    def __init__(self, model, category_titles_list: list):
        super().__init__(model)
        self.prompt_template = self._build_prompt_template(category_titles_list)

    def _build_prompt_template(self, category_titles_list: list) -> str:
        if category_titles_list is None:
            raise ValueError("No category_titles provided")

        category_titles_text = ", ".join([f"{c}" for c in category_titles_list])
        return classification_prompt.format(category_list=category_titles_text)

    def predict(self, item):
        prompt = self.prompt_template.format(ad_text=item)

        model_result = self.model.predict(
            prompt
        )

        return self.process_model_result(model_result)
