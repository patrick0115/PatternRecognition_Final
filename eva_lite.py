import tensorflow as tf
import numpy as np
from data_aggregation import data_aggregation as da
from tensorflow.keras.models import load_model
from tflite_model_maker.image_classifier import DataLoader
#train_set, test_set = da()
data = DataLoader.from_folder('test')
mydogcat = load_model('resources/mydogcat.h5')
# mydogcat_lite= load_model('resources/mydogcat(Lite).tflite')

mydogcat.evaluate_tflite('resources/mydogcat(Lite).tflite', data)

# def evaluate_tflite_model(classifier):
#   # Initialize TFLite interpreter using the model.
#   interpreter = tf.lite.Interpreter(model_content=classifier)
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


# Evaluate the TF Lite float model. You'll find that its accurary is identical
# to the original TF (Keras) model because they are essentially the same model
# stored in different format.
# f = open('resources/mydogcat(Lite).tflite', "wb")
# float_accuracy = evaluate_tflite_model(f)
# print('Float model accuracy = %.4f' % float_accuracy)

