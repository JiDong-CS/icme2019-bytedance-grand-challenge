import numpy as np
import pandas as pd
import time
import os
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', '数据目录')
flags.DEFINE_string('track_name', None, 'track名称')

flags.mark_flag_as_required("track_name")

track_name = FLAGS.track_name
data_dir = FLAGS.data_dir

train_file = os.path.join(data_dir, 'final_%s_train.txt' % track_name)
test_file = os.path.join(data_dir, 'final_%s_test_no_anwser.txt' % track_name)

names = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device_id', 'create_time', 'video_duration']


tf.logging.info("============== loading data ==================")
df_train = pd.read_csv(train_file, sep='\t', names=names)
df_test = pd.read_csv(test_file, sep='\t', names=names)
df_all = pd.concat([df_train, df_test])

create_time = df_all[['item_id', 'create_time']].drop_duplicates('item_id')
create_time = create_time.set_index('item_id')

create_time.to_csv(os.path.join(data_dir, '%s_item_create_time.csv' % track_name))

tf.logging.info("============== successfully ==================")
