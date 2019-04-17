import pandas as pd
import tensorflow as tf
import os
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', '数据目录')
flags.DEFINE_string('track_name', None, 'track名称')

flags.mark_flag_as_required("track_name")

track_name = FLAGS.track_name
data_dir = FLAGS.data_dir

names = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'device_id', 'create_time', 'video_duration']
train_data = pd.read_csv(os.path.join(data_dir, 'final_%s_train.txt' % track_name), sep='\t', names=names)
test_data = pd.read_csv(os.path.join(data_dir, 'final_%s_test_no_anwser.txt' % track_name), sep='\t', names=names)

data = pd.concat([train_data, test_data])
del train_data, test_data
data = data.reset_index(drop=True)

count_names = ['count_%d' % i for i in range(28)]
count_feats = pd.read_csv(os.path.join(data_dir, '%s_count_feats.csv' % track_name), sep='\t', skiprows=1, names=count_names, dtype=np.float32)

count_feats = (count_feats - count_feats.min()) / (count_feats.max() - count_feats.min())
#count_feats = count_feats.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

data = pd.concat([data, count_feats], axis=1)

tf.logging.info("concat completely. write data to file")

data.iloc[:19622340].to_csv(os.path.join(data_dir, 'final_%s_train_count.txt' % track_name), float_format='%.6f', index=False, header=False)
data.iloc[19622340:].to_csv(os.path.join(data_dir, 'final_%s_test_count.txt' % track_name), float_format='%.6f', index=False, header=False)