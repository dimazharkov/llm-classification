from dependency_injector import containers, providers

from app.controllers.experiment_controller import ExperimentController
from app.core.use_cases.compare_category_pair import CompareCategoryPairUseCase
from app.core.use_cases.experiment_one import ExperimentOneUseCase
from app.core.use_cases.experiment_three import ExperimentThreeUseCase
from app.core.use_cases.experiment_two import ExperimentTwoUseCase
from app.core.use_cases.pairwise_classification import PairwiseClassificationUseCase
from app.core.use_cases.predict_five_categories import PredictFiveCategoriesUseCase
from app.infrastructure.evaluators.category_confidence_evaluator import CategoryConfidenceEvaluator
from app.infrastructure.evaluators.classification_evaluator import ClassificationEvaluator
from app.infrastructure.evaluators.pairwise_evaluator import PairwiseEvaluator
from app.infrastructure.llm_clients.gemini_client import GeminiClient
from app.infrastructure.persistence.json_saver import JsonSaver
from app.repositories.advert_file_repository import AdvertFileRepository
from app.repositories.category_file_repository import CategoryFileRepository
from app.repositories.category_pair_file_repository import CategoryPairFileRepository


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    llm_client = providers.Singleton(GeminiClient)
    classification_evaluator = providers.Singleton(ClassificationEvaluator)
    category_confidence_evaluator = providers.Singleton(CategoryConfidenceEvaluator)
    pairwise_evaluator = providers.Singleton(PairwiseEvaluator)

    json_saver = providers.Factory(JsonSaver, path=config.TARGET_FILE_PATH)
    advert_file_repo = providers.Factory(AdvertFileRepository, path=config.ADVERTS_FILE_PATH)
    category_file_repo = providers.Factory(CategoryFileRepository, path=config.CATEGORIES_FILE_PATH)
    category_pair_file_repo = providers.Factory(CategoryPairFileRepository, path=config.CATEGORIES_PAIR_FILE_PATH)

    experiment_controller = providers.Factory(
        ExperimentController,
        advert_repository=advert_file_repo,
        evaluator=classification_evaluator,
        saver=json_saver,
    )

    experiment_one = providers.Factory(ExperimentOneUseCase, llm=llm_client, category_repo=category_file_repo)

    experiment_two = providers.Factory(ExperimentTwoUseCase, llm=llm_client, category_repo=category_file_repo)

    predict_five_categories_use_case = providers.Factory(
        PredictFiveCategoriesUseCase,
        llm=llm_client,
        category_repo=category_file_repo,
    )

    compare_category_pair_use_case = providers.Factory(
        CompareCategoryPairUseCase,
        llm=llm_client,
        advert_repo=advert_file_repo,
        category_pair_repo=category_pair_file_repo,
    )

    pairwise_classification_use_case = providers.Factory(
        PairwiseClassificationUseCase,
        llm=llm_client,
        category_evaluator=pairwise_evaluator,
    )

    experiment_three = providers.Factory(
        ExperimentThreeUseCase,
        predict_five_categories_use_case=predict_five_categories_use_case,
        compare_category_pair_use_case=compare_category_pair_use_case,
        pairwise_classification_use_case=pairwise_classification_use_case,
    )
