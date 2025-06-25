import os
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras.layers import StringLookup
import matplotlib.pyplot as plt
import logging
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
from keras.saving import register_keras_serializable
from concurrent.futures import ProcessPoolExecutor, wait
import json

# ---------- CONFIGURATION ----------
# Set key hyperparameters and constants
image_width, image_height = 128, 32
batch_size = 64
padding_token = 99
epochs = 5
split_runs = 3
AUTOTUNE = tf.data.AUTOTUNE
tf.get_logger().setLevel('ERROR')

# ---------- LOGGING ----------
# Set up logging and suppress warnings
LOG_FILE = "script.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings("ignore")


# ---------- LOAD AND CLEAN DATA ----------
def load_and_preprocess_words(filepath="data/words.txt"):
    """Load and filter lines from the dataset, ignoring comments and erroneous entries."""
    with open(filepath, "r") as f:
        lines = [l for l in f.readlines() if not l.startswith("#") and "err" not in l]
    return lines


def parse_paths_and_labels(samples, base_image_path="data/words"):
    """Extract image paths and corresponding labels from dataset lines."""
    paths, labels = [], []
    for line in samples:
        image_id = line.split()[0]
        label = line.strip().split()[-1]
        part1, part2 = image_id.split("-")[:2]
        img_path = os.path.join(base_image_path, part1, f"{part1}-{part2}", f"{image_id}.png")
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            paths.append(img_path)
            labels.append(label)
    return paths, labels


def get_vocabulary(labels):
    """Generate a sorted character set and find the max label length."""
    charset = sorted(set("".join(labels)))
    return charset, max(len(label) for label in labels)


# ---------- DATASET PIPELINE ----------
def distortion_free_resize(image, img_size=(image_width, image_height)):
    """Resize image to target size while maintaining aspect ratio and apply padding."""
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    image = tf.pad(image, paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image


def prepare_dataset(image_paths, labels, char_to_num, max_len):
    """Build a batched, preprocessed dataset from image paths and text labels."""

    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = distortion_free_resize(image)
        return tf.cast(image, tf.float32) / 255.0

    def vectorize_label(label):
        label = char_to_num(tf.strings.unicode_split(label, "UTF-8"))
        return tf.pad(label, [[0, max_len - ops.shape(label)[0]]], constant_values=padding_token)

    def process(image_path, label):
        return {"image": preprocess_image(image_path), "label": vectorize_label(label)}

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(process, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).prefetch(AUTOTUNE).cache()


# ---------- MODEL ----------
@register_keras_serializable()
class CTCLayer(keras.layers.Layer):
    """Custom Keras layer to compute CTC loss."""

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """Compute and register CTC loss in the model."""
        batch_len = tf.shape(y_true)[0]
        input_len = tf.shape(y_pred)[1] * tf.ones((batch_len, 1), dtype=tf.int32)
        label_len = tf.shape(y_true)[1] * tf.ones((batch_len, 1), dtype=tf.int32)
        loss = self.loss_fn(y_true, y_pred, input_len, label_len)
        self.add_loss(loss)
        return y_pred


def build_model(vocab_size):
    """Construct and compile the CNN-RNN model for handwriting recognition."""
    input_img = keras.Input(shape=(image_width, image_height, 1), name="image")
    labels = keras.Input(name="label", shape=(None,))

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Reshape((image_width // 4, (image_height // 4) * 64))(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Dense(vocab_size + 2, activation="softmax")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)
    model = keras.Model(inputs=[input_img, labels], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam())
    return model


# ---------- METRICS ----------
def calculate_edit_distance(y_true, y_pred, max_len, num_to_char):
    """Calculate average edit distance between true and predicted labels."""
    sparse_labels = ops.cast(tf.sparse.from_dense(y_true), dtype=tf.int64)
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded = keras.ops.nn.ctc_decode(y_pred, sequence_lengths=input_len)[0][0][:, :max_len]
    sparse_preds = ops.cast(tf.sparse.from_dense(decoded), dtype=tf.int64)
    distances = tf.edit_distance(sparse_preds, sparse_labels, normalize=False)
    return tf.reduce_mean(distances).numpy()


# ---------- PLOTTING ----------
def plot_metrics(train_losses, val_losses, edit_distances, run_name):
    """Save and log plots of training loss, validation loss, and edit distance."""
    if not os.path.exists("plots"):
        os.makedirs("plots")

    epochs_range = range(len(train_losses))

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_path = f"plots/loss_curve_{run_name}.png"
    plt.savefig(loss_path)
    mlflow.log_artifact(loss_path)
    plt.close()

    plt.figure()
    plt.plot(epochs_range, edit_distances, label="Edit Distance", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Edit Distance")
    plt.title("Average Edit Distance")
    plt.legend()
    edit_path = f"plots/edit_curve_{run_name}.png"
    plt.savefig(edit_path)
    mlflow.log_artifact(edit_path)
    plt.close()


# ---------- TRAINING ----------
def train_run(split_id, all_lines):
    """Execute one training run on a data split, log metrics and artefacts to MLflow."""
    np.random.seed(5402 + split_id)
    np.random.shuffle(all_lines)
    train, val_test = train_test_split(all_lines, test_size=0.1, random_state=split_id)
    val, test = train_test_split(val_test, test_size=0.5, random_state=split_id)

    train_paths, train_labels = parse_paths_and_labels(train)
    val_paths, val_labels = parse_paths_and_labels(val)
    test_paths, test_labels = parse_paths_and_labels(test)

    with open(f"test_paths_split_{split_id + 1}.txt", "w") as f:
        for item in test_paths:
            f.write(f"{item.replace('\\', '/')}\n")
    mlflow.log_artifact(f"test_paths_split_{split_id + 1}.txt")

    logging.info(f"Train-validation-test split complete for split {split_id + 1}")

    vocab, max_len = get_vocabulary(train_labels)
    char_to_num = StringLookup(vocabulary=list(vocab), mask_token=None)
    num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    vocab_dict = {
        'char_to_num': {char: int(index) for index, char in enumerate(char_to_num.get_vocabulary())},
        'num_to_char': {int(index): char for index, char in enumerate(char_to_num.get_vocabulary())}
    }
    with open(f'vocab_{split_id + 1}.json', 'w') as f:
        json.dump(vocab_dict, f)
    mlflow.log_artifact(f'vocab_{split_id + 1}.json')

    train_ds = prepare_dataset(train_paths, train_labels, char_to_num, max_len)
    val_ds = prepare_dataset(val_paths, val_labels, char_to_num, max_len)

    model = build_model(len(vocab))

    dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    last_dense_layer = dense_layers[-1]

    input_layer = model.input
    output_layer = last_dense_layer.output
    prediction_model = keras.Model(inputs=input_layer, outputs=output_layer)

    logging.info(f"Model built, training started for split {split_id + 1}")
    train_losses, val_losses, edit_distances = [], [], []

    for epoch in range(epochs):
        hist = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
        val_loss = hist.history["val_loss"][0]
        train_loss = hist.history["loss"][0]

        y_true, y_pred = [], []
        for batch in val_ds:
            y_true.append(batch["label"])
            y_pred.append(prediction_model.predict([batch["image"], batch["label"]], verbose=0))

        edit_distance = calculate_edit_distance(tf.concat(y_true, axis=0), np.concatenate(y_pred), max_len, num_to_char)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("edit_distance", edit_distance, step=epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        edit_distances.append(edit_distance)
        logging.info(f"Epoch {epoch + 1} of {epochs} complete for split {split_id + 1}")

    logging.info(f"Model trained, plotting metrics for split {split_id + 1}")
    plot_metrics(train_losses, val_losses, edit_distances, f"split{split_id + 1}")
    mlflow.keras.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "HandwritingRecognizer")
    logging.info(f"Model logged and registered for split {split_id + 1}")


def train_and_log_experiment(split_id, all_lines):
    """Run the full training pipeline for one split and track in MLflow."""
    try:
        with mlflow.start_run(run_name=f"split-{split_id + 1}") as run:
            mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "split_id": split_id + 1})
            logging.info(f"Starting run {split_id + 1}")
            train_run(split_id, all_lines)
    except Exception as e:
        logging.error(f"Training failed for split {split_id + 1}")
        mlflow.log_param("failure_reason", str(e))


# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        all_lines = load_and_preprocess_words()
        mlflow.set_experiment("Handwriting Recognition")

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(train_and_log_experiment, split, all_lines) for split in range(split_runs)]
            wait(futures)
    except Exception as e:
        logging.error("Main training failed. Error: " + str(e))
