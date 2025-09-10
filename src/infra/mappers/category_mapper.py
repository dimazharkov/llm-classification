from src.core.domain.category import Category
from src.infra.schemas.category_schema import CategorySchema


def schema_to_category(schema: CategorySchema) -> Category:
    """Schema → Domain"""
    return Category(**schema.model_dump())


def category_to_schema(entity: Category) -> CategorySchema:
    """Domain → Schema"""
    return CategorySchema.model_validate(entity.__dict__)
