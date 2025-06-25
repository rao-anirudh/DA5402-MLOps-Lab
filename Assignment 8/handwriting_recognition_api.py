import mlflow.keras
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from io import BytesIO
from fastapi import FastAPI, UploadFile
import uvicorn
import keras
from keras.layers import Layer
from keras.saving import register_keras_serializable
from tensorflow.keras import backend as K
from tensorflow.keras.layers import StringLookup
import json


best_model = "models:/HandwritingRecognizer/1"
best_run_id = "69731366acdd4d14821652081078c9ff"
best_vocab = "vocab_2.json"

# Create FastAPI app
app = FastAPI()


@register_keras_serializable()
class CTCLayer(Layer):
    """
    A custom Keras layer for computing CTC loss during training.
    """
    def __init__(self, **kwargs):
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """
        Adds CTC loss to the model during training.
        """
        batch_len = tf.shape(y_true)[0]
        input_len = tf.shape(y_pred)[1] * tf.ones((batch_len, 1), dtype=tf.int32)
        label_len = tf.shape(y_true)[1] * tf.ones((batch_len, 1), dtype=tf.int32)
        loss = self.loss_fn(y_true, y_pred, input_len, label_len)
        self.add_loss(loss)
        return y_pred

    def get_config(self):
        """
        Returns config for serialization.
        """
        return super(CTCLayer, self).get_config()

    @classmethod
    def from_config(cls, config):
        """
        Creates layer from saved config.
        """
        return cls(**config)


def load_model(model_uri):
    """
    Loads a trained model from MLflow and returns the prediction model.
    """
    custom_objects = {'CTCLayer': CTCLayer}
    model = mlflow.keras.load_model(model_uri, custom_objects=custom_objects)

    # Get the last dense layer as output
    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    last_dense_layer = dense_layers[-1]

    return keras.Model(inputs=model.input, outputs=last_dense_layer.output)


def load_vocab_artifact(run_id: str, artifact_path: str = "vocab.json"):
    """
    Loads the character index mapping from an MLflow artifact.
    """
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    with open(local_path, "r") as f:
        vocab_data = json.load(f)

    # Convert string keys to integers
    num_to_char = {int(k): v for k, v in vocab_data["num_to_char"].items()}
    return num_to_char


# Load model and vocab at startup
model = load_model(best_model)
num_to_char = load_vocab_artifact(best_run_id, best_vocab)


def preprocess_image(file: bytes):
    """
    Prepares an image for the model: resize, normalize, and add batch dimension.
    """
    image = Image.open(io.BytesIO(file)).convert("L")
    image = np.array(image)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    image = tf.image.resize(image, (128, 32))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    return image


def decode_ctc_output(predictions, num_to_char):
    """
    Decodes model output into text using greedy CTC decoding.
    """
    sequence_length = tf.ones([predictions.shape[0]], dtype=tf.int32) * predictions.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(predictions, sequence_length)

    predicted_indices = decoded[0].numpy()[0]
    decoded_text = ''.join([num_to_char.get(i, '') for i in predicted_indices])

    return decoded_text.replace('[PAD]', '').replace('[UNK]', '').strip()


@app.post("/predict")
async def predict(image: UploadFile):
    """
    Receives an image, runs inference, and returns the predicted text.
    """
    image_bytes = await image.read()
    processed_image = preprocess_image(image_bytes)

    # Dummy labels to match model input shape
    dummy_labels = np.zeros((processed_image.shape[0], 10))

    prediction = model.predict([processed_image, dummy_labels])
    predicted_text = decode_ctc_output(prediction, num_to_char)

    return {"text": predicted_text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5050)
