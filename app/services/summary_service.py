from app.core.contracts.llm_client_contract import LLMClientContract


class SummaryService:
    def __init__(self, model: LLMClientContract):
        self.model = model

    def summarize_adverts(self, dataset):
        pass

    def summarize_categories(self, dataset):
        pass