# shogi-recognizer

## Prerequisite

- Python `>= 3.7`
  - Tensorflow `>= 2.0`


## Training

```
pip install -r requirements.txt
```


### Train

```
python transfer.py --data_dir <Dataset Directory>
python finetuning.py --data_dir <Dataset Directory>
```


## Convert to JS

```
tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model <Saved Directory> js/
```
