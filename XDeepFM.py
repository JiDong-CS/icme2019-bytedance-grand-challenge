import tensorflow as tf


def init_weights(params):
    embedding_size = params['embedding_size']

    weights = dict()
    weights_initializer = tf.glorot_normal_initializer(dtype=tf.float32)
    bias_initializer = tf.constant_initializer(dtype=tf.float32)

    names = ['feature', 'word', 'item', 'author', 'music', 'item_city', 'video', 'audio', 'item_uid', 'author_uid',
             'music_uid']
    for name in names:
        key = '%s_embeddings' % name
        vocabulary_size = params['%s_size' % name]
        weights[key] = tf.get_variable(
            name=key,
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[vocabulary_size, embedding_size])

        first_order_key = 'first_order_%s_weights' % name
        weights[first_order_key] = tf.get_variable(
            name=first_order_key,
            dtype=tf.float32,
            initializer=weights_initializer,
            shape=[vocabulary_size, 1])

    weights['fm_bias'] = tf.get_variable(name='fm_bias', dtype=tf.float32, initializer=bias_initializer, shape=[1])

    return weights


def embed(features, weights, params):
    """ embed raw feature to embedding vector"""
    first_order_output_list = []
    embedding_list = []

    names = ['feature', 'word', 'item', 'author', 'music', 'item_city', 'item_uid', 'author_uid', 'music_uid']
    for name in names:
        field_size = params['%s_field_size' % name]
        feature_ids = tf.reshape(features['%s_ids' % name], shape=[-1, field_size])  # 特征id
        feature_values = tf.reshape(features['%s_weights' % name], shape=[-1, field_size, 1])  # 特征值
        
        print(feature_values)
        first_order_weights = tf.nn.embedding_lookup(weights['first_order_%s_weights' % name], ids=feature_ids)
        embeddings = tf.nn.embedding_lookup(weights['%s_embeddings' % name], ids=feature_ids)

        if name == 'feature':
            first_order_output = tf.reduce_sum(tf.multiply(feature_values, first_order_weights), axis=2)
            embeddings = tf.multiply(feature_values, embeddings)
        else:
            first_order_output = tf.reduce_sum(tf.multiply(feature_values, first_order_weights), axis=1)
            embeddings = tf.reduce_sum(tf.multiply(feature_values, embeddings), axis=1, keepdims=True)

        first_order_output_list.append(first_order_output)
        embedding_list.append(embeddings)

    print(first_order_output_list)
    for name in ['video', 'audio']:
        feature_values = features['%s_weights' % name]
        first_order_output = tf.matmul(feature_values, weights['first_order_%s_weights' % name])
        embeddings = tf.reshape(tf.matmul(feature_values, weights['%s_embeddings' % name]),
                                shape=[-1, 1, params['embedding_size']])
        first_order_output_list.append(first_order_output)
        embedding_list.append(embeddings)

    first_order_output = tf.concat(first_order_output_list, axis=1)
    embeddings = tf.concat(embedding_list, axis=1)

    return first_order_output, embeddings


class XDeepFM:

    def __init__(self, use_dnn=True, use_cin=True, use_fm=False):
        self.use_dnn = use_dnn
        self.use_cin = use_cin
        self.use_fm = use_fm

    @staticmethod
    def model_fn(features, labels, mode, params):
        # parse params
        embedding_size = params['embedding_size']  # 字段嵌入大小
        learning_rate = params['learning_rate']  # 学习率
        field_size = params['feature_field_size'] + 10  # 字段数量
        hidden_units = params['hidden_units']  # 各隐藏层隐藏单元数
        use_dnn = params.get('use_dnn', True)  # 是否使用Deep Neural Network(DNN), 默认True
        use_cin = params.get('use_cin', True)  # 是否使用Compressed Interactive Network(CIN), 默认True
        use_fm = params.get('use_fm', False)  # 是否使用Factorization Machine(FM), 默认False
        # optimizer_used = params.get('optimizer', 'adagrad')
        dropout_rate = params.get('dropout_rate', 0.5)
        training = False
        if mode == tf.estimator.ModeKeys.TRAIN:
            training = True

        tf.logging.info(params)
        tf.logging.info('is_training: ' + str(training))

        weights = init_weights(params)  # 初始化权重
        first_order_outputs, embeddings = embed(features, weights, params)
        last_layer_to_concat = [first_order_outputs]

        # FM part
        if use_fm:
            second_order_outputs = 0.5 * tf.subtract(tf.square(tf.reduce_sum(embeddings, axis=1)),
                                                     tf.reduce_sum(tf.square(embeddings), axis=1))
            last_layer_to_concat.append(second_order_outputs)

        # CIN Layer
        if use_cin:
            cin_layer_size = params.get('cin_layer_size', [10, 10, 10])
            field_nums = [field_size]
            cin_layer_0 = tf.split(embeddings, embedding_size * [1], 2)
            cin_layer_mat = [cin_layer_0]
            cin_layer_output = []
            for idx, layer_size in enumerate(cin_layer_size):
                conv_len = field_nums[0] * field_nums[-1]
                cross_result = tf.matmul(cin_layer_0, cin_layer_mat[-1], transpose_b=True)
                cross_result = tf.reshape(cross_result, shape=[embedding_size, -1, conv_len])
                cross_result = tf.transpose(cross_result, perm=[1, 0, 2])

                filters = tf.get_variable(name='filter_%d' % idx,
                                          shape=[1, conv_len, layer_size],
                                          initializer=tf.glorot_uniform_initializer(),
                                          dtype=tf.float32)
                b = tf.get_variable('cin_b_%d' % idx, shape=[layer_size], initializer=tf.zeros_initializer(),
                                    dtype=tf.float32)
                conv_result = tf.nn.conv1d(cross_result, filters, stride=1, padding='VALID')
                conv_result = tf.nn.relu(tf.nn.bias_add(conv_result, b))
                conv_result = tf.transpose(conv_result, perm=[0, 2, 1])

                cin_layer_mat.append(tf.split(conv_result, embedding_size * [1], 2))
                cin_layer_output.append(conv_result)
                field_nums.append(layer_size)

            cin_layer_output = tf.reduce_sum(tf.concat(cin_layer_output, axis=1), axis=2)
            cin_weights = tf.get_variable('cin_weights', dtype=tf.float32, shape=[cin_layer_output.shape[1], 40],
                                          initializer=tf.glorot_normal_initializer())
            cin_bias = tf.get_variable('cin_bias', dtype=tf.float32, shape=[40], initializer=tf.zeros_initializer())
            cin_layer_output = tf.nn.xw_plus_b(cin_layer_output, cin_weights, cin_bias)
            cin_layer_output = tf.layers.dropout(cin_layer_output, rate=dropout_rate, training=training)
            last_layer_to_concat.append(cin_layer_output)

        # DNN part
        if use_dnn:
            nn_outputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])

            for units in hidden_units:
                nn_outputs = tf.layers.dense(nn_outputs, units, activation=tf.nn.relu)
                nn_outputs = tf.layers.dropout(nn_outputs, rate=dropout_rate, training=training)

            last_layer_to_concat.append(nn_outputs)

        # Output layer
        outputs = tf.concat(last_layer_to_concat, axis=1)
        outputs = tf.layers.dense(outputs, 25, activation=tf.nn.relu)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        logits = tf.layers.dense(outputs, 1, activation=None)

        if mode == tf.estimator.ModeKeys.TRAIN:
            labels = tf.reshape(labels, shape=[-1, 1])  # 样本标签
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'probabilities': tf.sigmoid(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        if mode == tf.estimator.ModeKeys.EVAL:
            labels = tf.reshape(labels, shape=[-1, 1])  # 样本标签
            eval_metric_ops = {"auc": tf.metrics.auc(labels, tf.sigmoid(logits))}
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    from DataParser import DataParser

    dataParser = DataParser(track_name='track2', data_dir=None)
    features = {'feature_ids': tf.placeholder(dtype=tf.int32, shape=[32, 51]),
                'feature_weights': tf.placeholder(dtype=tf.float32, shape=[32, 51]),
                'video_weights': tf.placeholder(dtype=tf.float32, shape=[32, 128]),
                'audio_weights': tf.placeholder(dtype=tf.float32, shape=[32, 128]),
                'word_ids': tf.placeholder(dtype=tf.int32, shape=[32, 35]),
                'word_weights': tf.placeholder(dtype=tf.float32, shape=[32, 35]),
                'item_ids': tf.placeholder(dtype=tf.int32, shape=[32, 400]),
                'item_weights': tf.placeholder(dtype=tf.float32, shape=[32, 400]),
                'author_ids': tf.placeholder(dtype=tf.int32, shape=[32, 400]),
                'author_weights': tf.placeholder(dtype=tf.float32, shape=[32, 400]),
                'music_ids': tf.placeholder(dtype=tf.int32, shape=[32, 400]),
                'music_weights': tf.placeholder(dtype=tf.float32, shape=[32, 400]),
                'item_city_ids': tf.placeholder(dtype=tf.int32, shape=[32, 400]),
                'item_city_weights': tf.placeholder(dtype=tf.float32, shape=[32, 400]),
                'item_uid_ids': tf.placeholder(dtype=tf.int32, shape=[32, 150]),
                'item_uid_weights': tf.placeholder(dtype=tf.float32, shape=[32, 150]),
                'author_uid_ids': tf.placeholder(dtype=tf.int32, shape=[32, 500]),
                'author_uid_weights': tf.placeholder(dtype=tf.float32, shape=[32, 500]),
                'music_uid_ids': tf.placeholder(dtype=tf.int32, shape=[32, 500]),
                'music_uid_weights': tf.placeholder(dtype=tf.float32, shape=[32, 500])
                }

    labels = tf.placeholder(dtype=tf.float32, shape=[32, 1])
    print(dataParser.field_size)
    params = {
        'embedding_size': 40,
        'feature_field_size': dataParser.field_size,
        'feature_size': dataParser.feature_length,
        'hidden_units': [200, 100, 75, 50, 25],
        'cin_layer_size': [50, 50, 50, 50],
        'word_size': 134600,
        'word_field_size': 35,
        'item_size': 4122689,
        'item_field_size': 400,
        'author_size': 850308,
        'author_field_size': 400,
        'music_size': 89779,
        'music_field_size': 400,
        'item_city_size': 462,
        'item_city_field_size': 400,
        'video_size': 128,
        'audio_size': 128,
        'video_field_size': 128,
        'audio_field_size': 128,
        'item_uid_size': 73974,
        'item_uid_field_size': 150,
        'author_uid_size': 73974,
        'author_uid_field_size': 500,
        'music_uid_size': 73974,
        'music_uid_field_size': 500,
        'learning_rate': 0.005,
        'dropout_rate': 0.0,
        'batch_size': 32
    }

    estimatorSpec = XDeepFM.model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=params)
    print(estimatorSpec)
