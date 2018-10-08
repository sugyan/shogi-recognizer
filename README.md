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
  - Tensorflow `>= 1.8`
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
$ python ./script/train.py
```

### Freeze graph

```
$ python ./scripts/freeze_graph.py --labels logdir/labels.txt --checkpoint_path logdir/model.ckpt --output_graph output_graph.pb
```


### Convert to JS

```
$ tensorflowjs_converter --input_format tf_frozen_model --output_node_names 'MobilenetV2/Logits/output,labels' output_graph.pb ./js
```
