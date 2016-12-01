import pandas as pd




x = pd.read_csv('../raw_data/train.csv',delimiter=',', header=None, skiprows=1)
print(x.dtypes)
out = []
err_counter = 0

for index, row in x.iterrows():
    temp = [0]*10
    temp[0] = row[0]
    if pd.isnull(row[1]):
        out.append(temp)
        print(row)
        err_counter += 1
        continue

    labels = row[1].split(' ')
    for i in range(9):
        if str(i) in labels:
            temp[i+1] = 1
    out.append(temp)

out_frame = pd.DataFrame(out, columns=['business_id', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8',])
print(out_frame.shape)
print(err_counter)
print(out_frame.dtypes)
out_frame.sort_values('business_id', inplace=True)
out_frame.to_csv('bin_labels.csv',sep=',', index=False)

'''
0    1627
1     NaN
Name: 385, dtype: object
0    2661
1     NaN
Name: 928, dtype: object
0    2941
1     NaN
Name: 1083, dtype: object
0    430
1    NaN
Name: 1678, dtype: object
(1996, 10)
'''