from flask import Flask, request, send_file, abort
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow_hub as hub
from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from io import BytesIO

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
app_secret = 'dlwoKSLwWicmo47z9274hfSldkDKfj'

app = Flask(__name__)

# API:
# content - FILE
# style - FILE
# secret - STRING

@app.route('/styletransfer', methods=["POST"])
def style_transfer():
    received_app_secret = request.form['secret']
    if received_app_secret != app_secret:
        abort(401)

    content_tmp_image = request.files['content']
    style_tmp_image = request.files['style']

    content_image = plt.imread(content_tmp_image)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.

    style_image = plt.imread(style_tmp_image)
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))

    final_image_array = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    final_image = tf.keras.preprocessing.image.array_to_img(final_image_array[0])
    
    img_io = BytesIO()
    final_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')