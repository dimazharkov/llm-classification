import time
from collections.abc import Iterable

from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.domain.advert import Advert
from app.core.dto.advert_raw import AdvertRaw
from app.core.dto.category_prediction import AdvertCategoryPrediction
from app.core.use_cases.preprocess_adverts import PreprocessAdvertsUseCase
from app.core.use_cases.summarize_advert import SummarizeAdvertUseCase
from app.helpers.os_helper import load_from_disc, save_to_disc


class AdvertController:
    def preprocess(
        self,
        source_path: str,
        target_path: str,
        adverts_per_category: int = 20,
        categories_ids: Iterable[int] = None,
    ) -> None:
        raw = load_from_disc(source_path)
        parsed = [AdvertRaw.model_validate(ad) for ad in raw]
        processed = PreprocessAdvertsUseCase(parsed).run(categories_ids, adverts_per_category)

        payload = [item.model_dump(mode="json") for item in processed]
        save_to_disc(payload, target_path)

    def summarize(self, source_path: str, target_path: str, llm: LLMClientContract, rate_limit=1) -> None:
        raw = load_from_disc(source_path)
        parsed = [Advert.model_validate(ad) for ad in raw]

        use_case = SummarizeAdvertUseCase(llm)
        resumed = []

        for i, advert in enumerate(parsed, start=1):
            advert_with_resume = use_case.run(advert)
            resumed.append(advert_with_resume)
            print(".", end="")
            if i % 10 == 0:
                self._save_adverts(resumed, target_path)
            time.sleep(rate_limit)

        self._save_adverts(resumed, target_path)

    def indexing(self, source_path: str, target_path: str):
        raw = load_from_disc(source_path)
        adverts_list = [Advert.model_validate(ad) for ad in raw]

        indexed_list = []
        for idx, advert in enumerate(adverts_list):
            advert.advert_id = idx + 1
            advert.advert_summary = (advert.advert_summary or "").replace("\n", " ").replace("\r", " ").strip()
            indexed_list.append(advert)

        self._save_adverts(indexed_list, target_path)
        print("done")

    def analyze_results(self, source_path: str, rejected_path: str):
        try:
            experiment_results = load_from_disc(source_path)
        except FileNotFoundError:
            print("Achtung! source_path is empty!")
            return

        if not experiment_results:
            print("Empty results")
            return

        try:
            rejected_adverts = load_from_disc(rejected_path)
        except FileNotFoundError:
            rejected_adverts = {}

        stats = experiment_results.pop()
        prediction_list = [AdvertCategoryPrediction.model_validate(prediction) for prediction in experiment_results]

        for prediction in prediction_list:
            if prediction.tp:
                continue
            if prediction.advert_id:
                rejected_adverts[prediction.advert_id] = rejected_adverts.get(prediction.advert_id, 0) + 1

        save_to_disc(rejected_adverts, rejected_path)
        print("done")




    def _save_adverts(self, adverts: list[Advert], target_path: str) -> None:
        payload = [ad.model_dump(mode="json") for ad in adverts]
        save_to_disc(payload, target_path)

