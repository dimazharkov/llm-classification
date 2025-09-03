from dependency_injector import containers, providers

from app.resources.prompt_strategies.category_kw_prediction import CategoryKwPrediction
from app.resources.prompt_strategies.category_prediction import CategoryPrediction
from app.services.experiment_service import ExperimentService
from core.use_cases.experiment_one import ExperimentOneUseCase
from core.use_cases.experiment_two import ExperimentTwoUseCase
from infra.clients.llm.openai_client import OpenAIClient
from infra.clients.llm.prompt_registry import PromptRegistry
from infra.clients.llm.client_runner import LLMClientRunner
from infra.evaluators.classification_evaluator import ClassificationEvaluator
from infra.clients.llm.gemini_client import GeminiClient
from infra.repositories.advert_file_repository import AdvertFileRepository
from infra.repositories.category_file_repository import CategoryFileRepository
from infra.repositories.experiment_file_repository import ExperimentFileRepository


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    gemini_client = providers.Singleton(GeminiClient)
    openai_client = providers.Singleton(OpenAIClient)

    llm_client = providers.Selector(
        config.llm_client_name,
        openai=openai_client,
        gemini=gemini_client
    )

    prompt_strategies = providers.List(
        providers.Singleton(CategoryPrediction),
        providers.Singleton(CategoryKwPrediction),
        # providers.Singleton(AdvertSummarize),
        # providers.Singleton(CategoryDifference),
        # providers.Singleton(CategoryPairPrediction),
        # providers.Singleton(NCategoryKwPrediction)
    )

    strat_registry = providers.Factory(PromptRegistry, strategies=prompt_strategies)

    llm_runner = providers.Factory(LLMClientRunner, llm_client=llm_client, strat_registry=strat_registry)

    classification_evaluator = providers.Singleton(ClassificationEvaluator)
    # category_confidence_evaluator = providers.Singleton(CategoryConfidenceEvaluator)
    # pairwise_evaluator = providers.Singleton(PairwiseEvaluator)

    # json_saver = providers.Factory(JsonSaver, path=config.TARGET_FILE_PATH)
    advert_repo = providers.Factory(AdvertFileRepository, path=config.adverts_path)
    category_repo = providers.Factory(CategoryFileRepository, path=config.categories_path)
    # category_pair_file_repo = providers.Factory(CategoryPairFileRepository, path=config.CATEGORIES_PAIR_FILE_PATH)
    experiment_repo = providers.Factory(ExperimentFileRepository, path=config.target_path)

    experiment_service = providers.Factory(
        ExperimentService,
        advert_repo=advert_repo,
        experiment_repo=experiment_repo,
        evaluator=classification_evaluator
    )

    experiment_one = providers.Factory(ExperimentOneUseCase, llm_runner=llm_runner, category_repo=category_repo)
    experiment_two = providers.Factory(ExperimentTwoUseCase, llm_runner=llm_runner, category_repo=category_repo)
    #
    # predict_five_categories_use_case = providers.Factory(
    #     PredictFiveCategoriesUseCase,
    #     llm=llm_client,
    #     category_repo=category_file_repo,
    # )
    #
    # compare_category_pair_use_case = providers.Factory(
    #     CompareCategoryPairUseCase,
    #     llm=llm_client,
    #     advert_repo=advert_file_repo,
    #     category_pair_repo=category_pair_file_repo,
    # )
    #
    # pairwise_classification_use_case = providers.Factory(
    #     PairwiseClassificationUseCase,
    #     llm=llm_client,
    #     category_evaluator=pairwise_evaluator,
    # )
    #
    # experiment_three = providers.Factory(
    #     ExperimentThreeUseCase,
    #     predict_five_categories_use_case=predict_five_categories_use_case,
    #     compare_category_pair_use_case=compare_category_pair_use_case,
    #     pairwise_classification_use_case=pairwise_classification_use_case,
    # )
