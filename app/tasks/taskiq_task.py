# from taskiq import TaskiqDepends, task
# from app.controllers.advert_controller import AdvertController
# from app.infrastructure.providers.container import Container  # DI
#
# @task
# async def summarize_ads():
#     controller: AdvertController = Container.advert_controller()
#     controller.summarize_all()


# --- запуск воркера
# taskiq app.tasks.summarize_adverts_task:summarize_ads --broker redis://localhost:6379
