from sklearn.feature_extraction.text import TfidfVectorizer
import json
import time
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data', '数据目录')
flags.DEFINE_string('track_name', None, 'track名称')

flags.mark_flag_as_required("track_name")

track_name = FLAGS.track_name
data_dir = FLAGS.data_dir
title_file = os.path.join(data_dir, '%s_title.txt' % track_name)

def title_iter(title_file):
    with open(title_file) as file:
        for line in file:
            yield list(json.loads(line)['title_features'].keys())

tf.logging.info("\n============ start to generate word idf ===================")
start_time = time.time()

vectorizer = TfidfVectorizer(tokenizer=lambda line: line, lowercase=False)
vectorizer.fit(title_iter(title_file))

vocabulary = vectorizer.vocabulary_
idf = vectorizer.idf_

with open(os.path.join(data_dir, '%s_title_idf.csv' % track_name), 'w') as file:
    for word_id, index in vocabulary.items():
        file.write(word_id + ' ' + str(idf[index]) + '\n')

tf.logging.info("============= Completely. Take %f seconds ====================" % (time.time() - start_time))