import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import numpy as np
from tensorflow.keras.models import load_model
# from data_aggregation import data_aggregation as da

classifier = load_model('resources/dogcat_model_bak(Orginal).h5')
# classifier = load_model('resources/mydogcat.h5')

# Convert Keras model to TF Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(classifier )
tflite_float_model = converter.convert()


# Show model size in KBs.
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)


# Re-convert the model to TF Lite using quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Show model size in KBs.
quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)
print('which is about %d%% of the float model size.'\
      % (quantized_model_size * 100 / float_model_size))

# Save the quantized model to file to the Downloads directory
# f = open('resources/mydogcat(Lite).tflite', "wb")
f = open('resources/dogcat_model_bak(Lite).tflite', "wb")
f.write(tflite_quantized_model)
f.close()

print('`mnist.tflite` has been downloaded')


# # A helper function to evaluate the TF Lite model using "test" dataset.
# def evaluate_tflite_model(tflite_model):
#   # Initialize TFLite interpreter using the model.
#   interpreter = tf.lite.Interpreter(model_content=tflite_model)
#   interpreter.allocate_tensors()
#   input_tensor_index = interpreter.get_input_details()[0]["index"]
#   output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

#   # Run predictions on every image in the "test" dataset.
#   prediction_digits = []
#   for test_image in test_images:
#     # Pre-processing: add batch dimension and convert to float32 to match with
#     # the model's input data format.
#     test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
#     interpreter.set_tensor(input_tensor_index, test_image)

#     # Run inference.
#     interpreter.invoke()

#     # Post-processing: remove batch dimension and find the digit with highest
#     # probability.
#     digit = np.argmax(output()[0])
#     prediction_digits.append(digit)

#   # Compare prediction results with ground truth labels to calculate accuracy.
#   accurate_count = 0
#   for index in range(len(prediction_digits)):
#     if prediction_digits[index] == test_labels[index]:
#       accurate_count += 1
#   accuracy = accurate_count * 1.0 / len(prediction_digits)

#   return accuracy

# # Evaluate the TF Lite float model. You'll find that its accurary is identical
# # to the original TF (Keras) model because they are essentially the same model
# # stored in different format.
# float_accuracy = evaluate_tflite_model(tflite_float_model)
# print('Float model accuracy = %.4f' % float_accuracy)

# # Evalualte the TF Lite quantized model.
# # Don't be surprised if you see quantized model accuracy is higher than
# # the original float model. It happens sometimes :)
# quantized_accuracy = evaluate_tflite_model(tflite_quantized_model)
# print('Quantized model accuracy = %.4f' % quantized_accuracy)
# print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))


