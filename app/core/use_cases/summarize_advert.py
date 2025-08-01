from app.core.contracts.llm_client_contract import LLMClientContract
from app.core.contracts.use_case_contract import UseCaseContract
from app.core.domain.advert import Advert
from app.core.helpers.prompt_helper import format_prompt
from app.core.prompts.advert_summarize_prompt import advert_summarize_prompt


class SummarizeAdvertUseCase(UseCaseContract):
    def __init__(self, llm: LLMClientContract) -> None:
        self.llm = llm

    def run(self, advert: Advert) -> Advert:
        prompt = format_prompt(advert_summarize_prompt, title=advert.advert_title, text=advert.advert_text)

        summary = self.llm.generate(prompt)
        summarized_advert: Advert = advert.model_copy(update={"advert_summary": summary})

        return summarized_advert
