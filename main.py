import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
events_data = pd.read_csv('event_data_train.csv', encoding='windows-1251', sep=',')
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s') #перевод даты в читаемый вид
events_data['day'] = events_data.date.dt.date

events_data.groupby('day') \
    .user_id.nunique() \
    .plot() #в первом приближении проверяем данные на адекватность (визуально)