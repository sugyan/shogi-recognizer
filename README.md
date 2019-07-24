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
$ python ./scripts/save_model.py --labels logdir/labels.txt --checkpoint_path logdir/model.ckpt --output_graph ./output
```


## Convert to JS

```
$ tensorflowjs_converter --input_format tf_saved_model ./output ./js
```
