import pandas as pd
from tqdm import tqdm
from collections import Counter
import warnings
import gc
from multiprocessing import Pool

warnings.filterwarnings("ignore")

cols = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
        'finish', 'like', 'music_id', 'device', 'time', 'duration_time']
data_path = 'data'
train_path = os.path.join(data_path, 'final_track2_train.txt')
test_path = os.path.join(data_path, 'final_track2_test_no_anwser.txt')

# 开发时使用小数据集
# train_data = pd.read_csv(train_path, sep='\t', header=None, names=cols, nrows=200)
# test_data = pd.read_csv(test_path, sep='\t', header=None, names=cols, nrows=200)
train_data = pd.read_csv(train_path, sep='\t', header=None, names=cols)
test_data = pd.read_csv(test_path, sep='\t', header=None, names=cols)

def downcast_data(D):
    for c, d in zip(D.columns, D.dtypes):
        if d.kind == 'f':
            D[c] = pd.to_numeric(D[c], downcast='float')
        elif d.kind == 'i':
            D[c] = pd.to_numeric(D[c], downcast='signed')
    return D


data = pd.concat([train_data, test_data], axis=0)

del train_data
del test_data
gc.collect()

data = downcast_data(data)

data.head()

for col in data.columns:
    data[col] = data[col].astype(int)
    data.time = data.time - min(data.time)
data['time_day'] = data.time // (3600 * 24)


def count_fun(data, group_cols):
    new_feature = '_'.join(group_cols) + '_count'
    data[new_feature] = ""
    for c in group_cols:
        data[new_feature] = data[new_feature] + '_' + data[c].astype(str) 
    count = Counter(data[new_feature])
    data[new_feature] = data[new_feature].apply(lambda x: count[x])
    return data[[new_feature]]


count_feats_list = []
count_feats_list.append(['time_day'])

print('single feature count')
count_feats_list.extend([[c] for c in data.columns if c not in ['time', 'channel', 'like', 'finish']])

print('cross count')
users = ['uid']
# authors = ['user_city', 'author_id', 'item_city', 'channel', 'music_id', 'device', 'duration_time']
authors = ['item_id', 'user_city', 'author_id', 'item_city', 'channel', 'music_id', 'device', 'duration_time']
count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

users = ['author_id']
# authors = ['item_city', 'music_id', 'duration_time', 'time_day']
authors = ['channel', 'user_city', 'item_city', 'music_id', 'duration_time', 'time_day']
count_feats_list.extend([[u_col, a_col] for u_col in users for a_col in authors])

count_feats_list.append(['uid', 'user_city', 'channel', 'device'])
count_feats_list.append(['author_id', 'item_city', 'music_id', 'duration_time'])

# 根据自身配置修改 线程数
processor = 12
res_list = []

p = Pool(processor)

for i in range(len(count_feats_list)):
    res_list.append(p.apply_async(count_fun,
                                  args=(data[count_feats_list[i]], count_feats_list[i])))
    print(str(i) + ' processor started !')
p.close()
p.join()
print('done!')

res = pd.concat([res.get() for res in res_list], axis=1)

res.head()

# data = pd.concat([data, res], axis=1)

new_feats_list = ['_'.join(c) + '_count' for c in count_feats_list]

# data[new_feats_list].head()

res[new_feats_list].to_csv('data/track2_count_feats.csv', index=False, sep='\t')
