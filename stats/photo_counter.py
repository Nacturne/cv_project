import pandas as pd

x = pd.read_csv('../raw_data/train_photo_to_biz_ids.csv',delimiter=',')
group = x.groupby('business_id').size()


bin_label = pd.read_csv('bin_labels.csv', delimiter=',')
id_set = bin_label['business_id'].unique()
bin_label['photo_num'] = [0]*bin_label.shape[0]

for index, row in bin_label.iterrows():
    id = row['business_id']
    row['photo_num'] = group[id]

bin_label.to_csv('label_photo.csv',sep=',', index=False)



