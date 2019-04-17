import numpy as np
import pandas as pd
import time
import os
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from multiprocessing import Array, Pool, Manager

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', '数据目录')
flags.DEFINE_string('track_name', None, 'track名称')

flags.mark_flag_as_required("track_name")
# In[2]:


track_name = FLAGS.track_name
data_dir = FLAGS.data_dir

if track_name == 'track2':
    max_uid = 73974
    max_item_id = 4122689
    max_author_id = 850308
    max_music_id = 89778
    max_item_city = 461
elif track_name == 'track1':
    max_uid=663011
    max_item_id=31180491
    max_author_id=15595721
    max_music_id=7730983
    

train_file = os.path.join(data_dir, 'final_%s_train.txt' % track_name)
test_file = os.path.join(data_dir, 'final_%s_test_no_anwser.txt' % track_name)
names = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device_id', 'create_time', 'video_duration']

tf.logging.info("============== start to build behavior and audience feature =================")
tf.logging.info("============== loading data ==================")
df_train = pd.read_csv(train_file, sep='\t', names=names)
df_test = pd.read_csv(test_file, sep='\t', names=names)
df_all = pd.concat([df_train, df_test])
del df_train, df_test
tf.logging.info("============== load data successfully ==================")

# 统计每个视频\作者\音乐的观看人群
def group_iter(df, by_name, target_name):
    df_group = df[[by_name, target_name]].groupby(by_name)
    for i, g in df_group:
        yield g[target_name].tolist()
    
def fit_transform(df, by_name, target_name, save_dir, vocabulary):
    """
    df:数据帧
    by_name: 分组id名称
    target_name: 目标id名称
    save_dir: 存储目录
    vocabulary: 目标id的词汇表
    """
    vectorizer = TfidfVectorizer(tokenizer=lambda line: line, lowercase=False, vocabulary=vocabulary)
    mat = vectorizer.fit_transform(group_iter(df, by_name, target_name))
    scipy.sparse.save_npz(os.path.join(data_dir, '%s_%s_%s.npz' % (track_name, save_dir, target_name)), mat)
    
def convert_df(name, max_id, df):
    ids= df.drop_duplicates('%s_id' % name)
    ids_full = pd.DataFrame(np.zeros([max_id], dtype=np.int32), columns=['zeros']).drop(index=[0])
    if name=='music':
        print(ids_full)
    ids_join = ids.join(ids_full, on='%s_id' % name, how='right')
    ids_join = ids_join[ids_join['uid'].isna()].drop(columns=['zeros']).fillna(3).astype(np.int32)
    return pd.concat([df, ids_join])

def build_audience(name, max_id, df):
    try:
        tf.logging.info("============== building %s-audience feature =====================" % name)
        target_names = [('uid', max_uid)]
        df = convert_df(name, max_id, df)
        
        by_name='%s_id' % name # 按此id 分组
        save_dir = name
        for target_name, vocab_size in target_names:
            start_time = time.time()
            tf.logging.info("============= %s-audience 开始. 时间: %f ===============" % (name, start_time))

            vocabulary = dict([[x, x] for x in range(vocab_size)])
            fit_transform(df, by_name, target_name, save_dir, vocabulary)

            tf.logging.info("============= %s-audience 结束, Take %f seconds ===============" % (name, time.time()-start_time))
    except Exception as err:
        print(err)
        
pool = Pool(3)
name_ids = [('item', max_item_id), ('author', max_author_id), ('music', max_music_id)]
for name, max_id in name_ids:
    pool.apply_async(build_audience, args=(name, max_id, df_all))
    
pool.close()
pool.join()
tf.logging.info("All task done")