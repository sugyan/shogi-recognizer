import tfcoreml as tf_converter

tf_converter.convert(
    tf_model_path='output_graph.pb',
    mlmodel_path='Shogi.mlmodel',
    input_name_shape_dict={'input:0': [1, 128, 128, 3]},
    image_input_names=['input:0'],
    output_feature_names=['final_result:0'])
