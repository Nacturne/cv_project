import pandas as pd

x = pd.read_csv('label_photo.csv',delimiter=',')



data = x['photo_num']

print(data.min())
print(data.max())
print(data.mean())
print(data.median())
print(data.std())

