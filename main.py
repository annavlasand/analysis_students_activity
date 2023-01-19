import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
events_data = pd.read_csv('event_data_train.csv', encoding='windows-1251', sep=',')
submissions_data = pd.read_csv('submissions_data_train.csv', encoding='windows-1251', sep=',')

events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')#перевод даты в читаемый вид
events_data['day'] = events_data.date.dt.date
submissions_data['day'] = submissions_data.date.dt.date

events_data.groupby('day') \
    .user_id.nunique() \
    .plot() #в первом приближении проверяем данные на адекватность (визуально)

events_data.pivot_table(index='user_id',
                        columns='action',
                        values='step_id',
                        aggfunc='count',
                        fill_value=0).reset_index() #видоизменяем датафрейм, чтобы было понятно, сколько и какие шаги прошел юзер

users_scores = submissions_data.pivot_table(index='user_id',
                        columns='submission_status',
                        values='step_id',
                        aggfunc='count',
                        fill_value=0).reset_index() #смотрим, сколько было правильных и неправильных сабмитов

#так как нам нужно понять, кого именно считать покинувшими курс - тех, кто не появлялся на курсе 2 недели, месяц и тд,
# проводим небольшие дополнительные исследования

#рассчитываем какие были промежутки для каждого пользователя между прохождениями курса в днях:
gap_data = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']) \
    .groupby('user_id')['timestamp'].apply(list) \
    .apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data, axis=0))
gap_data = gap_data / (24 * 60 * 60)
quant_95 = gap_data.quantile(0.95)  # 5 % юзеров возвращаются на курс после примерно 2х месяцев
quant_90 = gap_data.quantile(0.9)   # 10 % юзеров возвращаются на курс после примерно 18 дней
