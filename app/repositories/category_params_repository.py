import copy
import json

from app.helpers import save_to_disc, load_from_disc


class CategoryParamsRepository:
    def __init__(self, file_name: str = "category_params.json"):
        self.file_name = file_name
        self._params = load_from_disc(self.file_name)

    def save_params(self, category_id: str, params) -> None:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string passed as params.")
        self._params[category_id] = params

    def get_params(self, category_id: str) -> str | None:
        # params = self._params.get(str(category_id))
        params = copy.deepcopy(self._params.get(str(category_id)))
        return params

    def __del__(self):
        try:
            save_to_disc(self._params, self.file_name)
        except Exception:
            pass  # безопасное удаление, без падения при завершении
