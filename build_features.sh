#!/bin/sh

# 默认数据目录均为./data，./data下没有次级目录，可通过--data_dir修改数据目录
# 前四个程序没有依赖关系，可同时执行
# DataParser.py 依赖 前四个程序 输出文件

# extract behavior and audience feature
python3 build_behavior.py --track_name=track2

python3 build_audience.py --track_name=track2

# generate word idf
python3 Word_IDF.py --track_name=track2

python3 item_create_time.py --track_name=track2

python3 ConcatFile.py --track_name=track2

python3 count_feats.py

# DataParser.py 依赖 前三个程序 输出文件
# generate feature files
python3 DataParser.py --track_name=track2 --chunk_size=400000 --num_process=50 # 对track2训练集　测试集　分块并生成tf_record格式特征文件
