# class DataPrepPipeline:
#     def __init__(self, repo, col_filter):
#         self.steps = [
#             CleanUseCase(...),
#             NormalizeUseCase(...),
#             AddPersonalDataUseCase(repository=repo),
#             AggregateDataUseCase(column_filter=col_filter),
#         ]
#
#     def run(self, df: pd.DataFrame) -> pd.DataFrame:
#         for step in self.steps:
#             df = step.run(df)
#         return df