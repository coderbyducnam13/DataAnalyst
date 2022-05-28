import pandas as pd
df=pd.read_csv(filepath_or_buffer = 'FoodPrice_in_Turkey.csv', sep = ',')
df['new_column'] = 'NaN'
df['Giảm giá']= pd.Series('10%', index=df.index)
df.insert(10,'Giảm giá 2',pd.Series('12%', index=df.index))
df=df.append({'Địa điểm':'NA','ProductId':'RR','Tên SP':'Rice','UmId':10,'UmName':'KG','Month':6,'Year':2021,'Price':84.3785,'Giảm giá':'10%','Giảm giá 2':'12%'},ignore_index=True)
df.tail()
df.pop('Giảm giá 2')
df.drop('Giảm giá', axis=1, inplace=True)
df.drop(['Month','Year'], axis=1, inplace=True)
df.drop(1, axis = 0, inplace=True)
drop()
df.drop([7377,7379], inplace=True)
df.head()
