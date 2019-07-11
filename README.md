# shogi-recognizer

## Prerequisite

- Python `>= 3.6`
  - Tensorflow `>= 1.12`


## Training

```
$ pip install -r requirements.txt
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
