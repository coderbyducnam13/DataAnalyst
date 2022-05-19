import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data\subset-covid-data.csv', encoding='UTF-8')
cleaned_data = data[data.date == '2020-04-12']
print ("trung bình số ca mắc mới: " + str(cleaned_data.cases.mean()))
print ("trung vị của số ca mắc mới: "+ str(cleaned_data.cases.median()))

plt.hist(cleaned_data.cases, bins = 200)
plt.title("Phân bố số ca mắc mới")
plt.xlabel("số số ca mắc mới")
plt.ylabel("Số lượng quốc gia")

print("tổng số ca nhiễm và số ca ncủa các châu lục")
cleaned_data.groupby('continent')['cases','deaths'].sum()

print ("5 quốc gia có số ca nhiễm mới cao nhất")
data = data.sort_values('cases',ascending = False)
print(data.head(5))

plt.show()
