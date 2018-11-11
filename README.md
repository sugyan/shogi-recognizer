# shogi-recognizer

## Image Resouces

- [Shogi images by muchonovski](http://mucho.girly.jp/bona/)
  - [Creative Commons 表示-非営利 2.1 日本 License](http://creativecommons.org/licenses/by-nc/2.1/jp/)
- [かわいいフリー素材集 いらすとや](https://www.irasutoya.com/)
- [しんえれ外部駒](http://shineleckoma.web.fc2.com/)
  - [Creative Commons 表示-非営利 2.1 日本 License](http://creativecommons.org/licenses/by-nc/2.1/jp/)
- [無料素材倶楽部](http://sozai.7gates.net/docs/japanese-chess/)


## Prerequisite

- Python `>= 3.6`
  - Tensorflow `>= 1.12`
  - Beautiful Soup `>= 4.6`
  - Pillow `>= 5.0`


## Training

```
$ pip install -r requirements.txt
```

### Prepare dataset

```
$ ./generate_dataset.sh
```

### Train

```
$ git submodule update --init
$ PYTHONPATH=./models/research/slim python ./trainer/task.py
```

#### Training on Cloud ML Engine

```
$ export BUCKET_NAME=<your bucket name>
$ ./cloudml/upload_data.sh
$ ./cloudml/run.sh
```

### Freeze graph

```
$ python ./scripts/freeze_graph.py --labels logdir/labels.txt --checkpoint_path logdir/model.ckpt --output_graph output_graph.pb
```


## Convert to JS

```
$ tensorflowjs_converter --input_format tf_frozen_model --output_node_names 'MobilenetV2/Logits/output,labels' output_graph.pb ./js
```
