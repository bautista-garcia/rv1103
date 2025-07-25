import tensorflow as tf
# 1 = float32 baseline, 0.35…1.4 = width multiplier
model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    alpha=1.0,                   # 1.0 = default width
    weights="imagenet",          # downloaded automatically
    include_top=True)            # keep classifier layer


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# # ---- optional INT8 path ----
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# def rep_ds():                                         # calibration set
#     for _ in range(128):
#         img = tf.random.uniform([1,224,224,3], 0, 255, tf.uint8)
#         yield [img]
# converter.representative_dataset = rep_ds
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type  = tf.uint8
# converter.inference_output_type = tf.uint8
# # -----------------------------

tflite_bytes = converter.convert()
open("mobilenet_v2.tflite","wb").write(tflite_bytes)
