from dependency_injector import containers, providers

from src.app.resources.prompt_strategies.category_difference import CategoryDifference
from src.app.resources.prompt_strategies.category_kw_prediction import CategoryKwPrediction
from src.app.resources.prompt_strategies.category_pair_prediction import CategoryPairPrediction
from src.app.resources.prompt_strategies.category_prediction import CategoryPrediction
from src.app.resources.prompt_strategies.n_category_kw_prediction import NCategoryKwPrediction
from src.app.services.category_service import CategoryService
from src.app.services.experiment_service import ExperimentService
from src.core.policies.evaluators.classification_evaluator import ClassificationEvaluator
from src.core.policies.evaluators.pairwise_evaluator import PairwiseEvaluator
from src.core.use_cases.compare_category_pair import CompareCategoryPairUseCase
from src.core.use_cases.experiment_one import ExperimentOneUseCase
from src.core.use_cases.experiment_three import ExperimentThreeUseCase
from src.core.use_cases.experiment_two import ExperimentTwoUseCase
from src.core.use_cases.pairwise_classification import PairwiseClassificationUseCase
from src.core.use_cases.predict_n_categories import PredictNCategoriesUseCase
from src.infra.clients.llm.client_runner import LLMClientRunner
from src.infra.clients.llm.gemini_client import GeminiClient
from src.infra.clients.llm.openai_client import OpenAIClient
from src.infra.clients.llm.prompt_registry import PromptRegistry
from src.infra.repositories.advert_file_repository import AdvertFileRepository
from src.infra.repositories.category_diff_file_repository import CategoryDiffFileRepository
from src.infra.repositories.category_file_repository import CategoryFileRepository
from src.infra.repositories.category_pair_file_repository import CategoryPairFileRepository
from src.infra.repositories.experiment_file_repository import ExperimentFileRepository
from src.infra.repositories.json_file_repository import JsonFileRepository


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    gemini_client = providers.Singleton(GeminiClient)
    openai_client = providers.Singleton(OpenAIClient)

    llm_client = providers.Selector(config.llm_client_name, openai=openai_client, gemini=gemini_client)

    prompt_strategies = providers.List(
        providers.Singleton(CategoryPrediction),
        providers.Singleton(CategoryKwPrediction),
        providers.Singleton(CategoryDifference),
        providers.Singleton(CategoryPairPrediction),
        providers.Singleton(NCategoryKwPrediction),
    )

    strat_registry = providers.Factory(PromptRegistry, strategies=prompt_strategies)

    llm_runner = providers.Factory(
        LLMClientRunner,
        llm_client=llm_client,
        strat_registry=strat_registry,
        delay_s=config.delay_s
    )

    classification_evaluator = providers.Singleton(ClassificationEvaluator)
    category_diff_repo = providers.Singleton(CategoryDiffFileRepository)
    pairwise_evaluator = providers.Singleton(PairwiseEvaluator)

    advert_repo = providers.Factory(AdvertFileRepository, path=config.adverts_path)
    category_repo = providers.Factory(CategoryFileRepository, path=config.categories_path)
    category_pair_repo = providers.Factory(CategoryPairFileRepository, path=config.categories_pairs_path)
    experiment_repo = providers.Factory(ExperimentFileRepository, path=config.target_path)
    file_repo = providers.Factory(JsonFileRepository, path=config.file_path)

    experiment_service = providers.Factory(
        ExperimentService,
        advert_repo=advert_repo,
        experiment_repo=experiment_repo,
        evaluator=classification_evaluator,
    )

    experiment_one = providers.Factory(ExperimentOneUseCase, llm_runner=llm_runner, category_repo=category_repo)
    experiment_two = providers.Factory(ExperimentTwoUseCase, llm_runner=llm_runner, category_repo=category_repo)

    predict_n_categories_use_case = providers.Factory(
        PredictNCategoriesUseCase,
        llm_runner=llm_runner,
        category_repo=category_repo,
    )
    compare_category_pair_use_case = providers.Factory(
        CompareCategoryPairUseCase,
        llm_runner=llm_runner,
        category_repo=category_repo,
        category_pair_repo=category_pair_repo,
        category_diff_repo=category_diff_repo,
    )
    pairwise_classification_use_case = providers.Factory(
        PairwiseClassificationUseCase,
        llm_runner=llm_runner,
        pairwise_evaluator=pairwise_evaluator,
    )
    experiment_three = providers.Factory(
        ExperimentThreeUseCase,
        predict_n_categories=predict_n_categories_use_case,
        compare_category_pair=compare_category_pair_use_case,
        pairwise_classification=pairwise_classification_use_case,
    )

    category_service = providers.Factory(
        CategoryService, category_repo=category_repo, advert_repo=advert_repo, file_repo=file_repo
    )
