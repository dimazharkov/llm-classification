from typing import Tuple


def format_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)

def parse_prediction_and_confidence(model_response: str) -> Tuple[str | None, float | None]:
    parts = [x.strip() for x in model_response.split(",")]

    if len(parts) == 2:
        predicted_title, confidence_str = parts
        try:
            confidence = float(confidence_str)
            return predicted_title, confidence
        except ValueError:
            return None, None

    return None, None
