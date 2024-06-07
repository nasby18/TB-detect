
# import cv2
# from pathlib import Path
# from PIL import Image
# import tensorflow as tf
# import numpy as np
# import tensorflow as tf
# import numpy as np
# from matplotlib.colors import Normalize
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
    
# def generate_grad_cam(model, img_path, output_path):
#     img = image.load_img(img_path, target_size=(128, 128))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer("conv2d_5").output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, np.argmax(predictions[0])]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     if len(conv_outputs.shape) == 1:
#         conv_outputs = tf.expand_dims(conv_outputs, axis=0)
#     elif len(conv_outputs.shape) == 2:
#         conv_outputs = tf.expand_dims(conv_outputs, axis=-1)
#     elif len(conv_outputs.shape) == 3:
#         conv_outputs = tf.expand_dims(conv_outputs, axis=0)
#     conv_outputs = conv_outputs * pooled_grads

#     heatmap = np.mean(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#     heatmap = np.uint8(255 * heatmap)

#     plt.imshow(img)
#     plt.imshow(heatmap, cmap='jet', alpha=0.4)
#     plt.axis('off')
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

import cv2
from pathlib import Path
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def generate_grad_cam(model, img_path, output_path):
    img_path = str(img_path)  # Ensure img_path is a string
    output_path = str(output_path)  # Ensure output_path is a string

    print(f"Generating Grad-CAM for: {img_path}, Output: {output_path}")  # Debug print

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer("conv2d_5").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # Remove the batch dimension
    conv_outputs = conv_outputs * pooled_grads[..., tf.newaxis]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(output_path, superimposed_img)
    print(f"Grad-CAM saved to: {output_path}")  # Debug print