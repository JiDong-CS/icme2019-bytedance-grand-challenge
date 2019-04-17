# coding: utf-8

# In[1]:
import tensorflow as tf
import os
import copy
from XDeepFM import XDeepFM
from DataParser import DataParser

task_name = None
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './data', '输入数据目录，根据任务类型分别读取train.txt、eval.txt、predict.txt')
flags.DEFINE_string('model_dir', None, '模型目录，用于模型保存与恢复')
flags.DEFINE_string('track_name', None, 'track名称')
flags.DEFINE_string('task_name', None, '任务名称，finish或者like')

flags.DEFINE_string('init_checkpoint_path', None, '初始化检查点路径')
flags.DEFINE_string('predict_output_path', None, '预测的输出路径')
flags.DEFINE_string('action', None, '执行动作，train、evaluate、predict或者train_evaluate')
flags.DEFINE_string('gpu_device', '0,1,2', 'GPU ID')

# model parameters
flags.DEFINE_string('model_name', 'xdeepfm', '模型名称，wide、deep、wide_deep、deep_fm、xdeepfm')
flags.DEFINE_float('learning_rate', 0.005, '学习率')
flags.DEFINE_float('dropout_rate', 0.5, 'dropout rate')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 1, 'epochs')
flags.DEFINE_integer('fm_embedding_size', 40, 'embedding size of the model which contains factorization machine part')

dataParser = None
data_dir = FLAGS.data_dir

train_filenames = [os.path.join('%s/%s_train_chunk_%d.tf_record' % (data_dir, FLAGS.track_name, i)) for i in range(35)]
eval_filenames = [os.path.join('%s/%s_train_chunk_%d.tf_record' % (data_dir, FLAGS.track_name, i)) for i in range(35, 50)]
test_filenames = [os.path.join('%s/%s_test_chunk_%d.tf_record' % (data_dir, FLAGS.track_name, i)) for i in range(7)]

def file_based_input_fn_builder(filenames, num_epochs, batch_size, is_training, params):
    """Creates an `input_fn` closure to be passed to Estimator."""

    global task_name, popped_name, dataParser
    name_to_features = {
        "finish": tf.FixedLenFeature([], tf.int64),
        "like": tf.FixedLenFeature([], tf.int64),
        "video_weights": tf.FixedLenFeature([], tf.string),
        "audio_weights": tf.FixedLenFeature([], tf.string),
    }

    feature_names = ['feature', 'word', 'item_uid', 'author_uid', 'music_uid']

    for name in feature_names:
        length = params['%s_field_size' % name]
        name_to_features['%s_ids' % name] = tf.FixedLenFeature([length], tf.int64)
        name_to_features['%s_weights' % name] = tf.FixedLenFeature([], tf.string)

    def get_user_behavior(uid):
        # user_behavior = dataParser.user_behavior[uid]
        # user_behavior_weights = dataParser.user_behavior_weights[uid]

        results = []
        for name in ['item', 'author', 'music', 'item_city']:
            results.append(dataParser.shared_arr_dict['user_%s_ids' % name][uid, :])
            results.append(dataParser.shared_arr_dict['user_%s_weights' % name][uid, :])
        return results

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            else:
                t = tf.decode_raw(t, out_type=tf.float32)

            example[name] = t

        results = tf.py_func(get_user_behavior, [example['feature_ids'][0]],
                             [tf.int32, tf.float32] * 4)

        example['item_ids'] = results[0]
        example['item_weights'] = results[1]
        example['author_ids'] = results[2]
        example['author_weights'] = results[3]
        example['music_ids'] = results[4]
        example['music_weights'] = results[5]
        example['item_city_ids'] = results[6]
        example['item_city_weights'] = results[7]

        label = tf.cast(example[task_name], tf.float32)
        return example, label

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(filenames=filenames)
        if is_training:
            d = d.shuffle(buffer_size=256)

        d = d.repeat(num_epochs)
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=1000
            ))

        return d

    return input_fn


def build_model(params=None):

    session_config = tf.ConfigProto()
    #session_config.gpu_options.allow_growth = True
    #session_config.gpu_options.allocator_type = 'BFC'
    run_config = tf.estimator.RunConfig().replace(
        save_checkpoints_secs=1500,# 每600秒保存检查点
        keep_checkpoint_max=2,
        session_config=session_config
    )

    estimator = None
    if FLAGS.model_name == 'deep_fm':
        params['use_cin'] = False
        params['use_fm'] = True
        estimator = tf.estimator.Estimator(
            model_fn=XDeepFM.model_fn,
            model_dir=FLAGS.model_dir,
            params=params,
            config=run_config
        )
    elif FLAGS.model_name == 'xdeepfm':
        estimator = tf.estimator.Estimator(
            model_fn=XDeepFM.model_fn,
            model_dir=FLAGS.model_dir,
            params=params,
            config=run_config
        )
    else:
        raise Exception('Unsupported model_name')

    return estimator


def train(params=None):
    train_input_fn = file_based_input_fn_builder(filenames=eval_filenames, num_epochs=FLAGS.num_epochs,
                                                 batch_size=FLAGS.batch_size, is_training=True, params=params)
    estimator = build_model(params)
    estimator.train(input_fn=train_input_fn)


def train_and_evaluate(params=None):
    train_input_fn = file_based_input_fn_builder(filenames=train_filenames, num_epochs=FLAGS.num_epochs,
                                                 batch_size=FLAGS.batch_size, is_training=True, params=params)
    eval_input_fn = file_based_input_fn_builder(filenames=eval_filenames, num_epochs=None,
                                                batch_size=512, is_training=False, params=params)

    estimator = build_model(params)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=2000)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    tf.logging.info('=================== Training is over. =====================')
    # evaluate the model on total validation set
    eval_input_fn = file_based_input_fn_builder(filenames=eval_filenames, num_epochs=1,
                                                batch_size=512, is_training=False, params=params)
    estimator.evaluate(input_fn=eval_input_fn, steps=12000)


def evaluate(params=None):
    eval_input_fn = file_based_input_fn_builder(filenames=eval_filenames, num_epochs=1,
                                                batch_size=2048, is_training=False, params=params)
    estimator = build_model(params)

    eval_result = estimator.evaluate(input_fn=eval_input_fn, steps=3000, checkpoint_path=FLAGS.init_checkpoint_path)
    print(eval_result)
    print('\nTest set accuracy: {auc:0.6f}\n'.format(**eval_result))


def predict(params=None):
    test_input_fn = file_based_input_fn_builder(filenames=test_filenames, num_epochs=1,
                                                batch_size=512, is_training=False, params=params)
    estimator = build_model(params)

    predictions = estimator.predict(input_fn=test_input_fn, checkpoint_path=FLAGS.init_checkpoint_path)
    count = 0
    if FLAGS.predict_output_path:
        with open(FLAGS.predict_output_path, 'w') as file:
            for prediction in predictions:
                prob_str = '%.6f' % prediction['probabilities'][0]
                file.write(prob_str + '\n')
                count += 1
        print(count)


def main(_):
    global dataParser, task_name, popped_name

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_device
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('model_name: %s' % FLAGS.model_name)
    tf.logging.info('data_dir: %s' % FLAGS.data_dir)
    tf.logging.info('model_dir: %s' % FLAGS.model_dir)
    tf.logging.info('task name: %s' % FLAGS.task_name)
    tf.logging.info('learning_rate: %s' % FLAGS.learning_rate)
    tf.logging.info('batch_size: %s' % FLAGS.batch_size)
    tf.logging.info('num_epochs: %s' % FLAGS.num_epochs)
    tf.logging.info('gpu_device: %s' % FLAGS.gpu_device)

    task_name = FLAGS.task_name
    if task_name == 'finish':
        popped_name = 'like'
    elif task_name == 'like':
        popped_name = 'finish'
    else:
        raise Exception('Only support finish or like task.')

    dataParser = DataParser(track_name=FLAGS.track_name, data_dir=FLAGS.data_dir)
    dataParser.load_user_behavior()
    # dataParser.load_conversion_rate()

    print("=========================== Feature Size: %d ============================" % dataParser.feature_length)
    params = None
    if FLAGS.model_name in ['deep_fm', 'xdeepfm']:
        params = {
            'embedding_size': FLAGS.fm_embedding_size,
            'feature_field_size': dataParser.field_size,
            'feature_size': dataParser.feature_length,
            'hidden_units': [200, 100, 75, 50, 25],
            'cin_layer_size': [50, 50, 50, 50],
            'word_size': dataParser.word_size,
            'word_field_size': 35,
            'item_size': dataParser.feature_dict['item_id'],
            'item_field_size': 400,
            'author_size': dataParser.feature_dict['author_id'],
            'author_field_size': 400,
            'music_size': dataParser.feature_dict['music_id'] + 1,
            'music_field_size': 400,
            'item_city_size': dataParser.feature_dict['item_city'] + 1,
            'item_city_field_size': 400,
            'video_size': 128,
            'audio_size': 128,
            'video_field_size': 128,
            'audio_field_size': 128,
            'item_uid_size': dataParser.feature_dict['uid'],
            'item_uid_field_size': 150,
            'author_uid_size': dataParser.feature_dict['uid'],
            'author_uid_field_size': 500,
            'music_uid_size': dataParser.feature_dict['uid'],
            'music_uid_field_size': 500,
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate,
            'batch_size': FLAGS.batch_size,
        }
        tf.logging.info('deep_fm params: ', params)

    if FLAGS.action == 'train':
        train(params)
    elif FLAGS.action == 'evaluate':
        evaluate(params)
    elif FLAGS.action == 'predict':
        tf.logging.info('predict_output_path: %s' % FLAGS.predict_output_path)
        predict(params)
    elif FLAGS.action == 'train_evaluate':
        train_and_evaluate(params)
    else:
        raise Exception(
            'The action %s is unsupported. Only support train, evaluate, predict, train_evaluate.' % FLAGS.action)


if __name__ == '__main__':
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("action")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("track_name")
    tf.app.run()
