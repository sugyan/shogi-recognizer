import tfcoreml as tf_converter

tf_converter.convert(
    tf_model_path='output_graph.pb',
    mlmodel_path='Shogi.mlmodel',
    input_name_shape_dict={'input:0': [1, 128, 128, 3]},
    output_feature_names=['final_result:0'],
    image_input_names=['input:0'],
    class_labels='output_labels.txt',
    red_bias=-1.0,
    green_bias=-1.0,
    blue_bias=-1.0,
    image_scale=2.0/255.0)
