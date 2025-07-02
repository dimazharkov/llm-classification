import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Чтение crontab-файла
crontab_path = Path("../static/cron.txt")
with crontab_path.open('r', encoding='utf-8') as f:
    crontab_lines = f.readlines()

daily_hour_stats = defaultdict(int)

# Разбор расписания
for line in crontab_lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    parts = re.split(r'\s+', line, maxsplit=5)
    if len(parts) < 6:
        continue

    minute_field, hour_field, day_of_month, month, day_of_week = parts[:5]

    # Фильтрация: оставляем только ежедневные задачи
    if day_of_month != '*' or day_of_week != '*':
        continue

    # Определяем количество запусков в час по полю минут
    if minute_field == '*':
        runs_per_hour = 60
    elif '/' in minute_field:  # например, */5
        step = int(minute_field.split('/')[1])
        runs_per_hour = 60 // step
    else:
        # Просто считаем количество минут, перечисленных через запятую
        runs_per_hour = len(minute_field.split(','))

    # Определяем часы
    if hour_field == '*':
        hours = list(range(24))
    else:
        hours = []
        for part in hour_field.split(','):
            if '/' in part:
                if '-' in part:
                    # Пример: 10-20/3
                    range_part, step = part.split('/')
                    start, end = map(int, range_part.split('-'))
                    step = int(step)
                    hours.extend(range(start, end + 1, step))
                else:
                    # Пример: */2
                    step = int(part.split('/')[1])
                    hours.extend(range(0, 24, step))
            elif '-' in part:
                start, end = map(int, part.split('-'))
                hours.extend(range(start, end + 1))
            else:
                hours.append(int(part))

    # Для всех часов, в которых выполняется задача, добавляем количество запусков
    for hour in hours:
        daily_hour_stats[hour] += runs_per_hour

# Создаем DataFrame для удобства
df = pd.DataFrame(sorted(daily_hour_stats.items()), columns=["Hour", "Daily Tasks"])

# Вывод
print(df.to_string(index=False))

# Линейный график
plt.figure(figsize=(10, 5))
plt.plot(df["Hour"], df["Daily Tasks"], marker='o')
plt.xlabel("Час (0–23)")
plt.ylabel("Количество запусков")
plt.title("Запуски ежедневных задач по часам (точный расчет)")
plt.grid(True)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()
