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

users_events_data = events_data.pivot_table(index='user_id',
                        columns='action',
                        values='step_id',
                        aggfunc='count',
                        fill_value=0).reset_index() #видоизменяем датафрейм, чтобы было понятно, сколько и какие шаги прошел юзер

users_scores = submissions_data.pivot_table(index='user_id',
                        columns='submission_status',
                        values='step_id',
                        aggfunc='count',
                        fill_value=0).reset_index() #смотрим попытки и баллы юзеров

#так как нам нужно понять, кого именно считать покинувшими курс - тех, кто не появлялся на курсе 2 недели, месяц и тд,
# проводим небольшие дополнительные исследования

#рассчитываем какие были промежутки для каждого пользователя между прохождениями курса в днях:
gap_data = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']) \
    .groupby('user_id')['timestamp'].apply(list) \
    .apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data, axis=0))
gap_data = gap_data / (24 * 60 * 60)
quant_95 = gap_data.quantile(0.95)  #5 % юзеров возвращаются на курс после примерно 2х месяцев
quant_90 = gap_data.quantile(0.9)   #10 % юзеров возвращаются на курс после примерно 18 дней

#возьмем приблизительно среднее - 30 дней и решим, что юзер считается покинувшим курс, если он не получил сертификат
#и не появлялся на курсе больше 30 дней

now = 1526772811 #день выгрузки данных
drop_out_threshold = 2592000 #30 дней
users_data = events_data.groupby('user_id', as_index=False) \
    .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold #условие для вылета с курса по отсутствию
users_data = users_data.merge(users_scores, on='user_id', how='outer')
users_data = users_data.fillna(0)
users_data = users_data.merge(users_events_data, how='outer')
users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index() #число уникальных дней для пользователей
users_data = users_data.merge(users_days, how='outer')
users_data.user_id.nunique() #проверка, не потеряли ли мы юзеров
events_data.user_id.nunique() #проверка, не потеряли ли мы юзеров - все ок
users_data['passed_corse'] = users_data.passed > 170 #условие окончания курса
#проанализируем поведение пользователя за его первые несколько дней, чтобы понять уйдет ли он с курса
#пробуем 3 дня
users_data[users_data.passed_corse].day.median()
user_min_time = events_data.groupby('user_id', as_index=False) \
    .agg({'timestamp': 'min'}) \
    .rename({'timestamp': 'min_timestamp'}, axis=1)
users_data = users_data.merge(user_min_time, how='outer')
events_data['user_time'] = events_data.user_id.map(str) + events_data.timestamp.map(str)
learning_time_threshold = 3 * 24 * 60 * 60
user_learning_time_threshold = user_min_time.user_id.map(str) + '_' + (user_min_time.min_timestamp + learning_time_threshold).map(str)
user_min_time['user_learning_time_threshold'] = user_learning_time_threshold
events_data = events_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')
events_data_train = events_data[events_data.user_time <= events_data.user_learning_time_threshold]

submissions_data['users_time'] = submissions_data.user_id.map(str) + '_' + submissions_data.timestamp.map(str)
submissions_data = submissions_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')
submissions_data_train = submissions_data[submissions_data.users_time <= submissions_data.user_learning_time_threshold]

X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index() \
    .rename(columns={'day': 'days'})
steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index() \
    .rename(columns={'step_id': 'steps_tried'})

X = X.merge(steps_tried, on='user_id', how='outer')

X = X.merge(submissions_data_train.pivot_table(index='user_id',
                        columns='submission_status',
                        values='step_id',
                        aggfunc='count',
                        fill_value=0).reset_index())
X['correct_ratio'] = X.correct  / (X.correct + X.wrong)

X = X.merge(events_data_train.pivot_table(index='user_id',
                        columns='action',
                        values='step_id',
                        aggfunc='count',
                        fill_value=0).reset_index()[['user_id', 'viewed']], how='outer')
X = X.fillna(0)