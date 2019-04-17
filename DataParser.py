import numpy as np
import json
import time
import tensorflow as tf
import pandas as pd
import ctypes
import math
import collections
import os
import gc
from multiprocessing import Array, Pool
import scipy.sparse

shared_arr = None
shared_arr_word_idf = None
shared_arr_title = None
shared_arr_face = None
shared_arr_time = None
shared_arr_video = None
shared_arr_audio = None


def parse_title_features(line):
    item = json.loads(line)
    item_id = item['item_id']
    title_features = item['title_features']
    if len(title_features) == 0:
        return

    words = []
    normalizer = 0
    for key, freq in title_features.items():
        word_id = np.int32(key)
        word_weight = freq * shared_arr_word_idf[word_id]
        normalizer += word_weight * word_weight
        words.append((word_id, word_weight))

    normalizer = math.sqrt(normalizer)
    # L2 norm
    words = sorted(map(lambda t: (t[0], t[1] / normalizer), words), key=lambda t: t[1], reverse=True)[:35]

    for _ in range(len(words), 35):
        words.append((0, 0.0))
    words = np.array(words)

    words[:, 1] = words[:, 1] / words[:, 1].sum()
    shared_arr_title[item_id, :] = words


def parse_face_features(line):
    item = json.loads(line)
    face_attrs = item['face_attrs']
    face_num = float(len(face_attrs))
    max_beauty = 0.0
    min_beauty = 0.0
    average_beauty = 0.0
    female_ratio = 0.0
    max_area = 0.0
    average_area = 0.0
    face_ratio = 0.0
    if face_num > 0:
        female_num = 0
        total_beauty = 0
        total_area = 0
        min_beauty = face_attrs[0]['beauty']
        for face in face_attrs:
            if face['beauty'] > max_beauty:
                max_beauty = face['beauty']
            if face['beauty'] < min_beauty:
                min_beauty = face['beauty']

            if face['gender'] == 0:
                female_num += 1
            total_beauty += face['beauty']

            relative_position = face['relative_position']
            area = relative_position[-1] * relative_position[-2]
            total_area += area
            if area > max_area:
                max_area = area
                face_ratio = relative_position[-2] / relative_position[-1]

        average_beauty = total_beauty / face_num
        female_ratio = female_num / face_num
        average_area = total_area / face_num

    shared_arr_face[item['item_id'], :] = [face_num, female_ratio, max_beauty, min_beauty, average_beauty, max_area,
                                           average_area, face_ratio]


def parse_sparse_vector(params):
    index = params[0]
    row = params[1]
    shape = params[2]
    name = params[3]
    words = zip(row.indices, row.data)
    words = np.array(sorted(words, key=lambda t: t[1], reverse=True)[:shape])

    if len(row.indices) > 0:
        words[:, 1] = words[:, 1] / words[:, 1].sum()
        shared_arr[index, :words.shape[0]] = words


def parse_video_embeddings(line):
    item = json.loads(line)
    if len(item['video_feature_dim_128']) == 128:
        item_id = item['item_id']
        shared_arr_video[item_id, :] = item['video_feature_dim_128']


def parse_audio_embeddings(line):
    item = json.loads(line)
    if len(item['audio_feature_128_dim']) == 128:
        item_id = item['item_id']
        shared_arr_audio[item_id, :] = item['audio_feature_128_dim']


def parse_time(item_time):
    item_id, seconds = item_time
    local_time = time.localtime(seconds)
    year = local_time.tm_year - 3649
    month = local_time.tm_mon - 1
    mday = local_time.tm_mday - 1
    wday = local_time.tm_wday
    hour = local_time.tm_hour
    minute = local_time.tm_min
    second = local_time.tm_sec
    month_day = month * 31 + mday

    hour_bin = hour // 6
    minute_bin = minute // 6
    season = month // 4
    y_m_d_h = 6 * (31 * (year * 12 + month) + mday) + hour_bin

    shared_arr_time[item_id, :] = [year, month, mday, wday, hour, minute, second, month_day, hour_bin, minute_bin,
                                   season, y_m_d_h]


class DataParser:
    column_names = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id',
                    'device_id', 'create_time', 'video_duration'] + ['count_%d' % i for i in range(28)]
    record_defaults = [np.int32] * 10 + [np.int64, np.int32] + [np.float32] * 28

    track2_feature_sizes = [('uid', 73974), ('user_city', 396), ('item_id', 4122689), ('author_id', 850308),
                     ('item_city', 461), ('music_id', 89778), ('device_id', 75085), ('channel', 5),
                     ('year', 4), ('month', 12), ('mday', 31), ('wday', 7), ('hour', 24), ('minute', 60),
                     ('month_day', 372), ('num_face', 1), ('female_ratio', 1), ('max_area', 1), ('avg_area', 1),
                     ('max_beauty', 1), ('min_beauty', 1), ('avg_beauty', 1), ('video_duration', 1)] + \
                           [('count_%d' % i, 1) for i in range(28) ]

    track1_feature_sizes = [('uid', 663011), ('user_city', 1), ('item_id', 31180491), ('author_id', 15595721),
                            ('item_city', 410), ('music_id', 7730983), ('device_id', 4246), ('channel', 5),
                            ('year', 4), ('month', 12), ('mday', 31), ('wday', 7), ('hour', 24), ('minute', 60),
                            ('month_day', 372), ('num_face', 1), ('female_ratio', 1), ('max_area', 1), ('avg_area', 1),
                            ('max_beauty', 1), ('min_beauty', 1), ('avg_beauty', 1), ('video_duration', 1)] + \
                           [('count_%d' % i, 1) for i in range(28) ]

    feature_offsets = {'uid': np.int32(0)}
    default_video_embedding = None
    default_audio_embedding = None
    default_face_features = None
    default_title_features = None
    default_statistics = {'num_item': 1, 'num_finish': 0, 'num_like': 0}

    feature_length = 0
    field_size = 0
    face_feature_size = 8
    time_feature_size = 12
    video_embedding_size = 128
    audio_embedding_size = 128
    max_title_length = 35
    word_size= 134600

    field_names = ['num_face', 'female_ratio', 'max_beauty', 'min_beauty',
                   'avg_beauty', 'max_area', 'avg_area', 'face_ratio'] + \
                  ['year', 'month', 'mday', 'wday', 'hour', 'minute', 'second',
                   'month_day', 'hour_bin', 'minute_bin', 'season', 'y_m_d_h'] + \
                  ['word_ids', 'word_weights'] + \
                  ['video_weights', 'audio_weights']  # + \

    # ['num_viewed', 'finish_0', 'finish_1', 'finish_nan', 'like_0', 'like_1', 'like_nan']

    def __init__(self, track_name, data_dir):
        self.track_name = track_name
        self.data_dir = data_dir

        if track_name == 'track1':
            self.feature_sizes = self.track1_feature_sizes
            self.word_size = 202687
        else:
            self.feature_sizes = self.track2_feature_sizes
            self.word_size = 134600

        self.feature_dict = dict(self.feature_sizes)
        self.features_is_loaded = False
        self.shared_arr_dict = dict()
        self.user_behavior = None
        self.user_behavior_weights = None
        self.viewers_info = dict()

        self.parse_field()
        self.init_data()

    def init_data(self):
        self.default_video_embedding = [np.float32(0.0) for _ in range(self.video_embedding_size)]
        self.default_audio_embedding = [np.float32(0.0) for _ in range(self.audio_embedding_size)]
        self.default_face_features = [np.float32(0.0) for _ in range(self.face_feature_size)]
        self.default_title_features = [(0.0, 0.0) for _ in range(self.max_title_length)]

    def parse_field(self):
        for i in range(1, len(self.feature_sizes)):
            cur_name, _ = self.feature_sizes[i]
            pre_name, size = self.feature_sizes[i - 1]
            self.feature_offsets[cur_name] = np.int32(self.feature_offsets[pre_name] + size + 1)
        self.feature_length = self.feature_offsets[self.feature_sizes[-1][0]] + \
                            self.feature_sizes[-1][1]
        self.field_size = len(self.feature_sizes)

    def load_features(self):

        if not self.features_is_loaded:
            self.load_audience_feature()
            self.load_title_features()
            self.load_face_features()
            self.load_time_features()
            self.load_video_embeddings()
            self.load_audio_embeddings()

            self.features_is_loaded = True

        return self.shared_arr_dict

    def load_user_behavior(self):
        global shared_arr

        if self.track_name == 'track1':
            name_shapes = {'item': 400, 'author': 200, 'music': 200, 'item_city': 200}
        else:
            name_shapes = {'item': 400, 'author': 400, 'music': 400, 'item_city': 400}

        for name, shape in name_shapes.items():
            tf.logging.info("=============== loading user_%s_behavior info ===================" % name)

            np_file = os.path.join(self.data_dir, "%s_user_%s_tf.npy" % (self.track_name, name))
            if os.path.exists(np_file):
                shared_arr = np.load(np_file).astype(np.float32)
            else:
                start_time = time.time()

                shared_arr_base = Array(ctypes.c_float, self.feature_dict['uid'] * shape * 2)
                shared_arr = np.frombuffer(shared_arr_base.get_obj(), dtype=np.float32)
                shared_arr = shared_arr.reshape(self.feature_dict['uid'], shape, 2)

                sparse_tfidf = scipy.sparse.load_npz(os.path.join(self.data_dir, '%s_user_%s_id.npz' % (self.track_name, name)))

                def row_iter():
                    for j, row in enumerate(sparse_tfidf):
                        yield (j, row, shape, name)

                pool = Pool(56)
                pool.map(parse_sparse_vector, row_iter())
                pool.close()

                np.save(np_file, shared_arr)
                end_time = time.time()
                tf.logging.info("take %f seconds" % (end_time - start_time))

            self.shared_arr_dict['user_%s_ids' % name] = shared_arr[:, :, 0].astype(np.int32)
            self.shared_arr_dict['user_%s_weights' % name] = shared_arr[:, :, 1].astype(np.float32)
            del shared_arr
            gc.collect()
            tf.logging.info("=============== load user_%s_behavior info successfully ===================" % name)

    def load_audience_feature(self):
        global shared_arr

        if self.track_name == 'track1':
            name_shapes = {'item': (self.feature_dict['item_id'], 150), 'author': (self.feature_dict['author_id'], 300),
                           'music': (self.feature_dict['music_id'], 300)}
        else:
            name_shapes = {'item': (self.feature_dict['item_id'], 150), 'author': (self.feature_dict['author_id'], 500),
                           'music': (self.feature_dict['music_id'], 500)}

        for name, shape in name_shapes.items():
            tf.logging.info("=============== loading %s_uid_tfidf info ===================" % name)

            np_file = os.path.join(self.data_dir, "%s_%s_uid_tfidf.npy" % (self.track_name, name))
            if os.path.exists(np_file):
                shared_arr = np.load(np_file).astype(np.float32)
            else:
                start_time = time.time()

                shared_arr_base = Array(ctypes.c_float, shape[0] * shape[1] * 2)
                shared_arr = np.frombuffer(shared_arr_base.get_obj(), dtype=np.float32)
                shared_arr = shared_arr.reshape(shape[0], shape[1], 2)

                sparse_tfidf = scipy.sparse.load_npz(os.path.join(self.data_dir, '%s_%s_uid.npz' % (self.track_name, name)))

                def row_iter():
                    for j, row in enumerate(sparse_tfidf):
                        yield (j, row, shape[1], None)

                pool = Pool(56)
                pool.map(parse_sparse_vector, row_iter())
                pool.close()

                np.save(np_file, shared_arr)

                end_time = time.time()
                tf.logging.info("take %f seconds" % (end_time - start_time))

            self.shared_arr_dict['%s_uid_ids' % name] = shared_arr[:, :, 0].astype(np.int32)
            self.shared_arr_dict['%s_uid_weights' % name] = shared_arr[:, :, 1].astype(np.float32)

            del shared_arr
            gc.collect()

            tf.logging.info("=============== load %s_uid_tfidf info successfully ===================" % name)

    def load_word_idf(self):
        global shared_arr_word_idf
        tf.logging.info("=============== loading word idf info ===================")

        num_words = 134560
        idf_array_base = Array(ctypes.c_float, num_words)
        shared_arr_word_idf = np.frombuffer(idf_array_base.get_obj(), dtype=np.float32)

        np_file = os.path.join(self.data_dir, "%s_word_idf.npy" % self.track_name)
        if os.path.exists(np_file):
            shared_arr_word_idf = np.load(np_file).astype(np.float32)
        else:
            with open(os.path.join(self.data_dir, '%s_title_idf.csv' % self.track_name)) as file:
                for line in file:
                    word_name, word_idf = line.split(" ")
                    shared_arr_word_idf[np.int(word_name)] = np.float32(word_idf)

                np.save(np_file, shared_arr_word_idf)

            tf.logging.info('vocab 100 idf: %f' % shared_arr_word_idf[100])

        tf.logging.info("=============== load word idf info successfully ===================")

    def load_face_features(self):
        global shared_arr_face

        tf.logging.info("=============== loading face_attrs===================")

        np_file = os.path.join(self.data_dir, "%s_face.npy" % self.track_name)
        start_time = time.time()
        if os.path.exists(np_file):
            shared_arr_face = np.load(np_file).astype(np.float32)
        else:
            shared_arr_face_base = Array(ctypes.c_float, self.feature_dict['item_id'] * self.face_feature_size)
            shared_arr_face = np.frombuffer(shared_arr_face_base.get_obj(), dtype=np.float32)
            shared_arr_face = shared_arr_face.reshape(self.feature_dict['item_id'], self.face_feature_size)

            with open(os.path.join(self.data_dir, '%s_face_attrs.txt' % self.track_name)) as file:
                pool = Pool(processes=56)
                pool.map(parse_face_features, file)
                pool.close()
                np.save(np_file, shared_arr_face)

        end_time = time.time()

        self.shared_arr_dict['face'] = shared_arr_face
        tf.logging.info(
            "===============load face_attrs successfully - take %f seconds==============" % (end_time - start_time))
        return shared_arr_face

    def load_title_features(self):
        global shared_arr_title

        tf.logging.info("=============== loading title features===================")

        np_title_file = os.path.join(self.data_dir, "%s_title.npy" % self.track_name)

        start_time = time.time()
        if os.path.exists(np_title_file):
            shared_arr_title = np.load(np_title_file).astype(np.float32)
        else:
            self.load_word_idf()

            shared_arr_title_base = Array(ctypes.c_float, self.feature_dict['item_id'] * self.max_title_length * 2)
            shared_arr_title = np.frombuffer(shared_arr_title_base.get_obj(), dtype=np.float32)
            shared_arr_title = shared_arr_title.reshape(self.feature_dict['item_id'], self.max_title_length, 2)

            with open(os.path.join(self.data_dir, '%s_title.txt' % self.track_name)) as file:
                pool = Pool(processes=56)
                pool.map(parse_title_features, file)
                pool.close()

                np.save(np_title_file, shared_arr_title)

        end_time = time.time()
        self.shared_arr_dict['word_ids'] = shared_arr_title[:, :, 0].astype(np.int32)
        self.shared_arr_dict['word_weights'] = shared_arr_title[:, :, 1].astype(np.float32)

        del shared_arr_title
        gc.collect()

        tf.logging.info(
            "===============load title features successfully - take %f seconds==============" % (end_time - start_time))

    def load_time_features(self):
        global shared_arr_time

        tf.logging.info("=============== parse time features===================")

        np_file = os.path.join(self.data_dir, "%s_time.npy" % self.track_name)
        start_time = time.time()
        if os.path.exists(np_file):
            shared_arr_time = np.load(np_file)
        else:
            item_time_df = pd.read_csv(os.path.join(self.data_dir, '%s_item_create_time.csv' % self.track_name), index_col=0)
            tf.logging.info("=============== data file has been read completely===================")

            shared_arr_time_base = Array(ctypes.c_int32, self.feature_dict['item_id'] * self.time_feature_size)
            shared_arr_time = np.frombuffer(shared_arr_time_base.get_obj(), dtype=np.int32)
            shared_arr_time = shared_arr_time.reshape(self.feature_dict['item_id'], self.time_feature_size)

            def item_time_iter():
                for t in item_time_df.itertuples():
                    yield [t.Index, t.create_time]

            pool = Pool(processes=56)
            pool.map(parse_time, item_time_iter())
            pool.close()

            np.save(np_file, shared_arr_time)

        self.shared_arr_dict['time'] = shared_arr_time
        end_time = time.time()
        tf.logging.info(
            "=========== parse time features successfully - take %d seconds=========" % (end_time - start_time))

        return shared_arr_time

    def load_audio_embeddings(self):
        global shared_arr_audio

        tf.logging.info("=============== loading audio_embeddings===================")

        np_file = os.path.join(self.data_dir, "%s_audio.npy" % self.track_name)

        start_time = time.time()
        if os.path.exists(np_file):
            shared_arr_audio = np.load(np_file).astype(np.float32)
        else:
            if self.track_name == 'track1':
                audio_filenames = ['track1_audio_features_part%d.txt' % i for i in range(1, 5)]
            else:
                audio_filenames = ['track2_audio_features.txt']
            shared_arr_audio_base = Array(ctypes.c_float, self.feature_dict['item_id'] * self.audio_embedding_size)
            shared_arr_audio = np.frombuffer(shared_arr_audio_base.get_obj(), dtype=np.float32)
            shared_arr_audio = shared_arr_audio.reshape(self.feature_dict['item_id'], self.audio_embedding_size)

            for audio_filename in audio_filenames:
                with open(os.path.join(self.data_dir, audio_filename)) as file:
                    pool = Pool(processes=56)
                    pool.map(parse_audio_embeddings, file)
                    pool.close()

            np.save(np_file, shared_arr_audio)

        self.shared_arr_dict['audio_weights'] = shared_arr_audio
        end_time = time.time()
        tf.logging.info("===============load audio_embeddings successfully - take %f seconds==============" % (
                end_time - start_time))

        return shared_arr_audio

    def load_video_embeddings(self):
        global shared_arr_video

        tf.logging.info("=============== loading video_embeddings===================")
        np_file = os.path.join(self.data_dir, "%s_video.npy" % self.track_name)

        start_time = time.time()
        if os.path.exists(np_file):
            shared_arr_video = np.load(np_file).astype(np.float32)
        else:
            if self.track_name == 'track1':
                video_filenames = ['track1_video_features_part%d.txt' % i for i in range(1, 12)]
            else:
                video_filenames = ['track2_video_features.txt']

            shared_arr_video_base = Array(ctypes.c_float, self.feature_dict['item_id'] * self.video_embedding_size)
            shared_arr_video = np.frombuffer(shared_arr_video_base.get_obj(), dtype=np.float32)
            shared_arr_video = shared_arr_video.reshape(self.feature_dict['item_id'], self.video_embedding_size)

            for video_filename in video_filenames:
                with open(os.path.join(self.data_dir, video_filename)) as file:
                    pool = Pool(processes=56)
                    pool.map(parse_video_embeddings, file)
                    pool.close()
            np.save(np_file, shared_arr_video)

        self.shared_arr_dict['video_weights'] = shared_arr_video
        end_time = time.time()
        tf.logging.info("===============load video_embeddings successfully - take %f seconds==============" % (
                end_time - start_time))

        return shared_arr_video

    def convert_single_example(self, features):

        uid = features['uid']
        item_id = features['item_id']

        features['video_duration'] /= 100.0

        features['feature_ids'] = []
        features['feature_weights'] = []

        names = ['item', 'author', 'music']
        for name in names:
            index = features['%s_id' % name]
            if index < 0:
                index = 0
                features['%s_uid_ids' % name] = self.shared_arr_dict['%s_uid_ids' % name][index]
                features['%s_uid_weights' % name] = np.zeros([len(features['%s_uid_ids' % name]), ], dtype=np.float32)
            else:
                features['%s_uid_ids' % name] = self.shared_arr_dict['%s_uid_ids' % name][index]
                features['%s_uid_weights' % name] = self.shared_arr_dict['%s_uid_weights' % name][index]

            # features['%s_ids' % name] = self.shared_arr_dict['user_%s_ids' % name][uid]
            # features['%s_weights' % name] = self.shared_arr_dict['user_%s_weights' % name][uid]

        features['word_ids'] = self.shared_arr_dict['word_ids'][item_id]
        features['word_weights'] = self.shared_arr_dict['word_weights'][item_id]

        features['video_weights'] = self.shared_arr_dict['video_weights'][item_id]
        features['audio_weights'] = self.shared_arr_dict['audio_weights'][item_id]

        face_feature = self.shared_arr_dict['face'][item_id]
        time_feature = self.shared_arr_dict['time'][item_id]

        features.update(dict(zip(['num_face', 'female_ratio', 'max_beauty', 'min_beauty', 'avg_beauty', 'max_area',
                                  'avg_area', 'face_ratio'], face_feature)))
        features.update((dict(zip(['year', 'month', 'mday', 'wday', 'hour', 'minute', 'second', 'month_day',
                                   'hour_bin', 'minute_bin', 'season', 'y_m_d_h'], time_feature))))

        for name, size in self.feature_sizes:
            if size > 1:
                features['feature_ids'].append(self.feature_offsets[name] + features[name])
                features['feature_weights'].append(np.float32(1.0))
            else:
                features['feature_ids'].append(self.feature_offsets[name])
                features['feature_weights'].append(np.float32(features[name]))

        features['feature_ids'] = np.array(features['feature_ids'], dtype=np.int32)
        features['feature_weights'] = np.array(features['feature_weights'], dtype=np.float32)
        return features

    def file_based_convert_examples_to_features(self, chunk, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
            return f

        features = collections.OrderedDict()

        int_feature_names = ['feature_ids', 'word_ids', 'item_uid_ids',
                             'author_uid_ids', 'music_uid_ids', 'finish', 'like']
        float_feature_names = ['feature_weights', 'video_weights', 'audio_weights', 'word_weights',
                               'item_uid_weights', 'author_uid_weights', 'music_uid_weights']

        writer = tf.python_io.TFRecordWriter(output_file)

        count = 1
        for row in chunk.itertuples():
            if count % 20000 == 0:
                tf.logging.info("Writing example %d" % count)
                writer.flush()

            count += 1

            raw_feature = dict(row._asdict())
            feature = self.convert_single_example(raw_feature)

            for name in int_feature_names:
                value = feature[name]
                if name in ['finish', 'like']:
                    value = [value]

                features[name] = create_int_feature(value)

            for name in float_feature_names:
                value = feature[name].tostring()
                features[name] = create_float_feature(value)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


if __name__ == "__main__":
    global cat

    flags = tf.flags
    FLAGS = flags.FLAGS

    flags.DEFINE_string('data_dir', 'data', '数据目录')
    flags.DEFINE_string('track_name', None, 'track名称')
    flags.DEFINE_integer('chunk_size', None, 'chunk大小')
    flags.DEFINE_integer('num_process', 32, '进程数')

    tf.logging.set_verbosity(tf.logging.INFO)
    flags.mark_flag_as_required("track_name")
    flags.mark_flag_as_required("chunk_size")

    track_name = FLAGS.track_name
    data_dir = FLAGS.data_dir
    dataParser = DataParser(track_name='track2', data_dir=data_dir)
    dataParser.load_features()

    def to_tf_record(args):
        global cat

        index = args[0]
        chunk = args[1]
        output_file = os.path.join(data_dir, '%s_%s_chunk_%d.tf_record' % (track_name, cat, index))

        dataParser.file_based_convert_examples_to_features(
            chunk=chunk,
            output_file=output_file)


    names = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id',
             'device_id', 'create_time', 'video_duration'] + ['count_%d' % i for i in range(28)]

    record_defaults = [np.int32] * 10 + [np.int64, np.int32] + [np.float32] * 28
    dtype = dict(zip(names, record_defaults))

    def chunk_iter(filename, chunk_size):
        df = pd.read_csv(filename, names=names, iterator=True, dtype=dtype)
        index = -1
        while True:
            try:
                index += 1
                yield (index, df.get_chunk(chunk_size))
            except:
                break

    for t in ['train', 'test']:
        cat = t
        if cat == 'train':
            filename = os.path.join(data_dir, 'final_%s_%s_count.txt' % (track_name, cat))
        else:
            filename = os.path.join(data_dir, 'final_%s_%s_count.txt' % (track_name, cat))

        pool = Pool(FLAGS.num_process)
        pool.map(to_tf_record, chunk_iter(filename, chunk_size=FLAGS.chunk_size))
        pool.close()
