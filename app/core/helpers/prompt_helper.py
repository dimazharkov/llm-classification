def format_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)


def parse_prediction(model_response: str) -> str | None:
    # если в ответе есть запятая — старый формат
    parts = [x.strip() for x in model_response.split(",")]
    if len(parts) == 2:
        return parts[0]  # игнорируем confidence
    elif len(parts) == 1:
        return parts[0]
    return None


def parse_prediction_and_confidence(
    model_response: str,
) -> tuple[str | None, float | None]:
    parts = [x.strip() for x in model_response.split(",")]

    if len(parts) == 2:
        predicted_title, confidence_str = parts
        try:
            confidence = float(confidence_str)
            return predicted_title, confidence
        except ValueError:
            return None, None

    return None, None
