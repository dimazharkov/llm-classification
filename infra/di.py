from dependency_injector import containers, providers

from app.resources.prompt_strategies.category_difference import CategoryDifference
from app.resources.prompt_strategies.category_kw_prediction import CategoryKwPrediction
from app.resources.prompt_strategies.category_pair_prediction import CategoryPairPrediction
from app.resources.prompt_strategies.category_prediction import CategoryPrediction
from app.resources.prompt_strategies.n_category_kw_prediction import NCategoryKwPrediction
from app.services.experiment_service import ExperimentService
from core.use_cases.compare_category_pair import CompareCategoryPairUseCase
from core.use_cases.experiment_one import ExperimentOneUseCase
from core.use_cases.experiment_three import ExperimentThreeUseCase
from core.use_cases.experiment_two import ExperimentTwoUseCase
from core.use_cases.pairwise_classification import PairwiseClassificationUseCase
from core.use_cases.predict_n_categories import PredictNCategoriesUseCase
from core.policies.evaluators.pairwise_evaluator import PairwiseEvaluator
from infra.clients.llm.client_runner import LLMClientRunner
from infra.clients.llm.gemini_client import GeminiClient
from infra.clients.llm.openai_client import OpenAIClient
from infra.clients.llm.prompt_registry import PromptRegistry
from core.policies.evaluators.classification_evaluator import ClassificationEvaluator
from infra.repositories.advert_file_repository import AdvertFileRepository
from infra.repositories.category_diff_file_repository import CategoryDiffFileRepository
from infra.repositories.category_file_repository import CategoryFileRepository
from infra.repositories.category_pair_file_repository import CategoryPairFileRepository
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
        providers.Singleton(CategoryDifference),
        providers.Singleton(CategoryPairPrediction),
        providers.Singleton(NCategoryKwPrediction)
    )

    strat_registry = providers.Factory(PromptRegistry, strategies=prompt_strategies)

    llm_runner = providers.Factory(LLMClientRunner, llm_client=llm_client, strat_registry=strat_registry)

    classification_evaluator = providers.Singleton(ClassificationEvaluator)
    category_diff_repo = providers.Singleton(CategoryDiffFileRepository)
    pairwise_evaluator = providers.Singleton(PairwiseEvaluator)

    advert_repo = providers.Factory(AdvertFileRepository, path=config.adverts_path)
    category_repo = providers.Factory(CategoryFileRepository, path=config.categories_path)
    category_pair_repo = providers.Factory(CategoryPairFileRepository, path=config.categories_pairs_path)
    experiment_repo = providers.Factory(ExperimentFileRepository, path=config.target_path)

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
    #
    compare_category_pair_use_case = providers.Factory(
        CompareCategoryPairUseCase,
        llm_runner=llm_runner,
        category_repo=category_repo,
        category_pair_repo=category_pair_repo,
        category_diff_repo=category_diff_repo,
        rate_limit=config.rate_limit
    )
    #
    pairwise_classification_use_case = providers.Factory(
        PairwiseClassificationUseCase,
        llm_runner=llm_runner,
        pairwise_evaluator=pairwise_evaluator,
        rate_limit=config.rate_limit
    )
    #
    experiment_three = providers.Factory(
        ExperimentThreeUseCase,
        predict_n_categories=predict_n_categories_use_case,
        compare_category_pair=compare_category_pair_use_case,
        pairwise_classification=pairwise_classification_use_case
    )
