from core.contracts.llm_client import LLMClient
from core.contracts.use_case_contract import UseCaseContract
from core.domain.advert import Advert
from core.policies.prompt_helper import format_prompt
from app.resources.prompt_strategies.advert_summarize_prompt import advert_summarize_prompt


class SummarizeAdvertUseCase(UseCaseContract):
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, advert: Advert) -> Advert:
        prompt = format_prompt(advert_summarize_prompt, title=advert.advert_title, text=advert.advert_text)

        summary = self.llm.generate(prompt)
        summarized_advert: Advert = advert.model_copy(update={"advert_summary": summary})

        return summarized_advert
