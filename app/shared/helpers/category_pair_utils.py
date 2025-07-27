from app.shared.types.category_id_pair import CategoryIdPair


def normalize_pair(pair: CategoryIdPair) -> CategoryIdPair:
    a, b = pair
    return (a, b) if a <= b else (b, a)


def make_pair_key(pair: CategoryIdPair) -> str:
    a, b = normalize_pair(pair)
    return f"{a}:{b}"


def split_pair_key(key: str) -> CategoryIdPair:
    a, b = key.split(":")
    return int(a), int(b)
