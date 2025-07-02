# from dependency_injector import containers, providers

# class Container(containers.DeclarativeContainer):
#     llm = providers.Singleton(OpenAIClient)
#     evaluator = providers.Singleton(RougeEvaluator)
#     summarize_service = providers.Factory(
#         SummaryService,
#         model=llm,
#         evaluator=evaluator
#     )