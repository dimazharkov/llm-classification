from core.domain.advert import Advert
from infra.schemas.advert_schema import AdvertSchema


def schema_to_advert(schema: AdvertSchema) -> Advert:
    """Schema → Domain"""
    return Advert(**schema.model_dump())

def advert_to_schema(entity: Advert) -> AdvertSchema:
    """Domain → Schema"""
    return AdvertSchema.model_validate(entity.__dict__)