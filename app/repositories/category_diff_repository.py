from app.helpers import save_to_disc, load_from_disc


class CategoryDiffRepository:
    def __init__(self, file_name: str = "category_diffs.json"):
        self.file_name = file_name
        self._diffs = load_from_disc(self.file_name)

    def _make_key(self, cat1: str, cat2: str) -> str:
        return "-".join(sorted([str(cat1), str(cat2)]))

    def save_difference(self, cat1: str, cat2: str, diff_text: str) -> None:
        key = self._make_key(cat1, cat2)
        self._diffs[key] = diff_text

    def get_difference(self, cat1: str, cat2: str) -> str | None:
        key = self._make_key(cat1, cat2)
        return self._diffs.get(key)

    def __del__(self):
        try:
            save_to_disc(self._diffs, self.file_name)
        except Exception:
            pass  # безопасное удаление, без падения при завершении
