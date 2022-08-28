import tensorflow as tf


def convert_tflite(tflite_model_path, save_model_path):

  converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
  tflite_model = converter.convert()

  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
