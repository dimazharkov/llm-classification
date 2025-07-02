from app.helpers import read_json


class CategoriesRepository():
    def __init__(self, source_file_name):
        self.categories = read_json(source_file_name)
        self.selected_ids = None

    def filter_by_ids(self, selected_ids: list = None, excluded_ids: list = None):
        if excluded_ids is None:
            excluded_ids = []

        if selected_ids is None:
            self.selected_ids = [k for k in self.categories.keys() if k not in excluded_ids]
        else:
            self.selected_ids = list(set(selected_ids) - set(excluded_ids))

        return self.selected_ids

    def get_category_titles(self, selected_ids: list = None):
        if selected_ids is None and self.selected_ids is None:
            raise AttributeError("You must specify at least one of selected_ids")

        if selected_ids is None:
            selected_ids = self.selected_ids

        return [self.categories[str(_id)]["category_title"] for _id in selected_ids]

    def get_by_id(self, category_id):
        return self.categories[str(category_id)]