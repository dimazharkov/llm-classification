import itertools
import random
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from app.ai.gemini import Gemini
from app.core.contracts.llm_client_contract import LLMClientContract
from app.evaluators.base import BaseEvaluator
from app.evaluators.direct_evaluator import DirectEvaluator
from app.evaluators.pairwise_evaluator import PairwiseEvaluator
from app.helpers.model_helper import save_interim_results
from app.helpers import load_from_disc
from app.helpers.prompt_helper import classification_prompt, category_selection_prompt
from app.helpers.text_helper import normalize_text, fix_latin_letters
from app.ai.openai import OpenAI

"""
0.173
0.069
0.129
"""
class ClassifierService:
    def __init__(self, model: LLMClientContract = None, request_timeout: int = 20):
        self.model = OpenAI(model_name="gpt-3.5-turbo") if model is None else model
        self.request_timeout = request_timeout

    def classify_direct(self, ads: list, category_titles: list) -> float:
        evaluator = DirectEvaluator(self.model, category_titles)
        return self._classify(ads, evaluator)

    def classify_pairwise(self, ads:list, category_params_repository, category_ids: list) -> float:
        evaluator = PairwiseEvaluator(self.model, category_params_repository, category_ids)
        return self._classify(ads, evaluator)

    def _classify(self, ads: list, evaluator: BaseEvaluator) -> float:
        evaluator.request_timeout = self.request_timeout
        y_true, y_pred = evaluator.evaluate(ads)

        f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
        return f1


class ClassifierService1:
    def __init__(self, data, categories):
        self.data = data
        self.categories = categories
        self.category_props = load_from_disc("category_params.json")
        self.wait_seconds = 20
        self.model = OpenAI(model_name="gpt-3.5-turbo")

    def run(self, selected_categories: list = None, excluded_categories: list = None):
        if excluded_categories is None:
            excluded_categories = []

        if selected_categories is None:
            selected_categories = [k for k in self.categories.keys() if k not in excluded_categories]
        else:
            selected_categories = list(set(selected_categories) - set(excluded_categories))

        selected_titles = [self.categories[str(_id)]["category_title"] for _id in selected_categories]

        shuffled_items = []
        for category_id in selected_categories:
            items = [item for item in self.data if int(item['category_id']) == int(category_id)]
            random.shuffle(items)
            shuffled_items.extend(items[:5])
        random.shuffle(shuffled_items)

        selected_titles_text = ", ".join([f"{c}" for c in selected_titles])
        formatted_prompt = classification_prompt.format(category_list=selected_titles_text)


        y_true = []
        y_pred = []
        y_pred_pw = []
        interim_results = []

        for i, item in enumerate(shuffled_items, 1):
            true_title = normalize_text(fix_latin_letters(
                item["category_title"]
            ))

            y_true.append(true_title)

            predicted_title, confidence = self.process(
                item["text"], formatted_prompt
            )

            y_pred.append(predicted_title)

            if predicted_title != true_title:
                print('.')
                predicted_title, confidence = self.pairwise_process(
                    item["text"], selected_categories
                )
                y_pred_pw.append(predicted_title)
            else:
                y_pred_pw.append(predicted_title)
                time.sleep(self.wait_seconds)

            print(f"true: {true_title}, pred: {predicted_title}, conf: {confidence}, TP: {true_title == predicted_title}")


            interim_results.append({
                "true": true_title,
                "pred": predicted_title,
                "conf": confidence,
                "TP": true_title == predicted_title,
            })

            if i % 10 == 0 or i == len(shuffled_items):
                save_interim_results(interim_results, "interim_results.json")

            if i == 30:
                break

        result = self.analyze(y_true, y_pred)
        result_pw = self.analyze(y_true, y_pred_pw)
        return result, result_pw

    def process(self, ad_text, prompt, model):
        formatted_prompt = prompt.format(ad_text=ad_text)

        model_result = model.predict(
            formatted_prompt
        )

        predicted_title, confidence = model_result.split(",")

        return normalize_text(fix_latin_letters(predicted_title)), confidence

    def analyze(self, y_true, y_pred):
        labels = list(set(y_true) | set(y_pred))

        encoder = LabelEncoder()
        encoder.fit(labels)

        y_true_encoded = encoder.transform(y_true)
        y_pred_encoded = encoder.transform(y_pred)

        f1 = f1_score(y_true_encoded, y_pred_encoded, average="macro")

        print(f"F1: {f1:.4f}")

        return f1

    def pairwise_process(self, ad_text, selected_categories, model):
        def run_pairwise(categories):
            score_sum = defaultdict(float)
            win_count = defaultdict(int)

            for cat1, cat2 in itertools.combinations(categories, 2):
                print(".")
                cat1 = str(cat1)
                cat2 = str(cat2)
                category_1 = self.categories[cat1]["category_title"]
                category_1_keywords = ", ".join(self.category_props[cat1]["keywords"])
                category_1_properties = ", ".join(self.category_props[cat1]["properties"])
                category_1_context = ", ".join(self.category_props[cat1]["context"])

                category_2 = self.categories[cat2]["category_title"]
                category_2_keywords = ", ".join(self.category_props[cat2]["keywords"])
                category_2_properties = ", ".join(self.category_props[cat2]["properties"])
                category_2_context = ", ".join(self.category_props[cat2]["context"])

                formatted_prompt = category_selection_prompt.format(
                    ad_text=ad_text,
                    category_1=category_1,
                    category_2=category_2,
                    category_1_properties=category_1_properties,
                    category_2_properties=category_2_properties,
                    category_1_context=category_1_context,
                    category_2_context=category_2_context,
                    category_1_keywords=category_1_keywords,
                    category_2_keywords=category_2_keywords,
                )

                model_result = model.predict(
                    formatted_prompt
                )
                # print(f"model_result: {model_result}")
                predicted_title, confidence = [x.strip() for x in model_result.split(",")]
                # print(f"pred: {predicted_title}, conf: {confidence}")

                score_sum[predicted_title] += float(confidence)
                win_count[predicted_title] += 1

                time.sleep(self.wait_seconds)

            return score_sum, win_count

        # Первый проход
        score_sum, win_count = run_pairwise(selected_categories)
        max_score = max(score_sum.values(), default=0)
        top_cats = [cat for cat, val in score_sum.items() if val == max_score]

        if len(top_cats) > 1:
            cat_values = np.array([score_sum[cat] for cat in top_cats])
            probs = np.exp(cat_values) / np.exp(cat_values).sum()
            final_idx = np.argmax(probs)
            final_category = top_cats[final_idx]
            confidence = float(probs[final_idx])
        else:
            final_category = top_cats[0]
            confidence = score_sum[final_category] / win_count[final_category]

        return final_category, round(confidence, 2)


# ------
# predicted_list = [normalize_text(fix_latin_letters(item)) for item in model_result.split(",")]
# if true_title in predicted_list:
#     predicted_title = true_title
# else:
#     predicted_title = predicted_list[0]

# predicted_title = normalize_text(fix_latin_letters(
#     predicted_title
# ))
