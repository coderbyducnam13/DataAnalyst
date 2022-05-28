import pandas as pd
data=pd.read_csv('data\OnlineRetail.csv', encoding = "ISO-8859-1")
country = data.Country.unique()
print ("số lượng các quốc gia: " + str(country.size))
print (data.info())
data['total'] = data['Quantity'] * data['UnitPrice'] 

# Giá trị đơn hàng của mỗi đơn hàng
total_invoices = data['total'].sum()
print ("số lượng hóa đơn bán ra: "+ str (total_invoices.size))
print ("Tổng doanh thu: " + str(total_invoices.sum()))

quantity_product = data.groupby(['StockCode', 'Description'])['Quantity'].sum().sort_values(ascending= False)
quantity_product.head(10)

# sử dụng groupby để tiến hành tính tổng (sum) theo nhóm.
#Vì mỗi mã hàng sẽ có một mô tả tương ứng, để rõ ràng, tiến hành gom nhóm theo 2 thuộc tính: StockCode, Description
quantity_product = data.groupby(['StockCode', 'Description'])['total'].sum().sort_values(ascending= False)
quantity_product.head(10)
data.head()
