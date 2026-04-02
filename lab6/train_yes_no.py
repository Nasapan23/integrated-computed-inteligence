from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras import layers, models


DATASET_URL = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
DATASET_CLASS_NAMES = ["yes", "no"]
MODEL_LABELS = ["no", "yes"]
SAMPLE_PLOT_LABELS = ["yes", "no"]
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
OUTPUT_SEQUENCE_LENGTH = 16000
SAMPLE_RATE = 16000
FRAME_LENGTH = 255
FRAME_STEP = 128
NUM_FRAMES = 124
NUM_FREQ_BINS = 128
EPOCHS = 50
SEED = 0
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a TinyML yes/no speech recognizer.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "mini_speech_commands",
        help="Path to the mini_speech_commands dataset directory.",
    )
    parser.add_argument(
        "--keras-model",
        type=Path,
        default=project_dir / "yes_no_model.keras",
        help="Path for the exported Keras model.",
    )
    parser.add_argument(
        "--tflite-model",
        type=Path,
        default=project_dir / "yes_no_model_quant.tflite",
        help="Path for the quantized TFLite model.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=project_dir / "no.wav",
        help="Local WAV file used for the sample inference plot.",
    )
    return parser.parse_args()


def ensure_dataset(data_dir: Path) -> Path:
    if (data_dir / "yes").is_dir() and (data_dir / "no").is_dir():
        return data_dir

    project_dir = data_dir.parent
    zip_path = project_dir / "mini_speech_commands.zip"
    print(f"Downloading dataset to {zip_path}...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    print(f"Extracting dataset into {project_dir}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(project_dir)

    return data_dir


def ensure_demo_audio(data_dir: Path, sample_file: Path) -> Path:
    if sample_file.exists():
        return sample_file

    source_files = sorted((data_dir / "no").glob("*.wav"))
    if not source_files:
        raise FileNotFoundError(f"No 'no' samples were found under {data_dir / 'no'}")

    source_file = source_files[0]
    shutil.copyfile(source_file, sample_file)
    return sample_file


def squeeze_audio(audio: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return tf.squeeze(audio, axis=-1), tf.cast(1 - label, tf.int32)


def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    spectrogram = tf.signal.stft(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., :NUM_FREQ_BINS]
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def make_spectrogram_ds(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    dataset = dataset.map(
        lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    return dataset.prefetch(AUTOTUNE)


def build_model(input_shape: tuple[int, int, int], norm_layer: layers.Normalization) -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Resizing(16, 16),
            norm_layer,
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ]
    )


def save_history_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(
    labels_true: np.ndarray, labels_pred: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    matrix = confusion_matrix(labels_true, labels_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names).plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def maybe_display_audio(waveform: np.ndarray, sample_rate: int) -> None:
    try:
        from IPython.display import Audio, display  # type: ignore

        if "JPY_PARENT_PID" in os.environ or "ipykernel" in sys.modules:
            display(Audio(waveform, rate=sample_rate))
    except Exception:
        pass


def load_wav(sample_file: Path) -> np.ndarray:
    sample_rate, samples = wavfile.read(sample_file)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz sample rate, found {sample_rate} Hz in {sample_file}")

    samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = samples[:, 0]

    max_abs = np.max(np.abs(samples)) if samples.size else 1.0
    if max_abs > 1.0:
        samples = samples / 32768.0

    if samples.shape[0] < OUTPUT_SEQUENCE_LENGTH:
        samples = np.pad(samples, (0, OUTPUT_SEQUENCE_LENGTH - samples.shape[0]))
    else:
        samples = samples[:OUTPUT_SEQUENCE_LENGTH]

    return samples


def save_sample_prediction_plot(probabilities: np.ndarray, output_path: Path) -> None:
    probability_map = dict(zip(MODEL_LABELS, probabilities.tolist()))
    ordered_probabilities = np.array([probability_map[label] for label in SAMPLE_PLOT_LABELS], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(SAMPLE_PLOT_LABELS, ordered_probabilities * 100.0, color=["#2563eb", "#ea580c"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Prediction for no.wav")
    for bar, prob in zip(bars, ordered_probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{prob * 100.0:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def representative_dataset(train_spectrogram_ds: tf.data.Dataset):
    for spectrogram, _ in train_spectrogram_ds.take(100):
        for index in range(spectrogram.shape[0]):
            yield [tf.cast(spectrogram[index : index + 1], tf.float32)]


def convert_to_tflite(model: tf.keras.Model, train_spectrogram_ds: tf.data.Dataset, output_path: Path) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(train_spectrogram_ds)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return tflite_model


def inspect_tflite_model(tflite_model: bytes, output_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ops = [op["op_name"] for op in interpreter._get_ops_details()]

    lines = [
        f"input_shape={input_details[0]['shape'].tolist()}",
        f"input_dtype={input_details[0]['dtype']}",
        f"output_shape={output_details[0]['shape'].tolist()}",
        f"output_dtype={output_details[0]['dtype']}",
        "ops=" + ", ".join(ops),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nTFLite model summary:")
    for line in lines:
        print(f"  {line}")


def collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels: list[np.ndarray] = []
    for _, batch_labels in dataset:
        labels.append(batch_labels.numpy())
    return np.concatenate(labels, axis=0)


def main() -> None:
    args = parse_args()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    data_dir = ensure_dataset(args.data_dir)
    sample_file = ensure_demo_audio(data_dir, args.sample_file)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
        subset="both",
        class_names=DATASET_CLASS_NAMES,
    )

    train_waveform_ds = train_ds.map(squeeze_audio, num_parallel_calls=AUTOTUNE)
    val_waveform_ds = val_ds.map(squeeze_audio, num_parallel_calls=AUTOTUNE)

    test_waveform_ds = val_waveform_ds.shard(num_shards=2, index=0)
    val_waveform_ds = val_waveform_ds.shard(num_shards=2, index=1)

    train_spectrogram_ds = make_spectrogram_ds(train_waveform_ds, training=True)
    val_spectrogram_ds = make_spectrogram_ds(val_waveform_ds, training=False)
    test_spectrogram_ds = make_spectrogram_ds(test_waveform_ds, training=False)

    example_spectrograms, _ = next(iter(train_spectrogram_ds.take(1)))
    input_shape = tuple(example_spectrograms.shape[1:])
    print(f"Model input shape: {input_shape}")

    norm_layer = layers.Normalization()
    norm_layer.adapt(train_spectrogram_ds.map(lambda spectrogram, _: spectrogram))

    model = build_model(input_shape, norm_layer)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    test_loss, test_accuracy = model.evaluate(test_spectrogram_ds, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    args.keras_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.keras_model)

    history_plot = args.keras_model.parent / "training_history.png"
    save_history_plot(history, history_plot)

    test_labels = collect_labels(test_spectrogram_ds)
    test_predictions = model.predict(test_spectrogram_ds, verbose=0)
    predicted_labels = np.argmax(test_predictions, axis=1)
    confusion_plot = args.keras_model.parent / "confusion_matrix.png"
    save_confusion_matrix_plot(test_labels, predicted_labels, MODEL_LABELS, confusion_plot)

    sample_waveform = load_wav(sample_file)
    maybe_display_audio(sample_waveform, SAMPLE_RATE)
    sample_spectrogram = get_spectrogram(tf.convert_to_tensor(sample_waveform, dtype=tf.float32))
    sample_prediction = model.predict(sample_spectrogram[tf.newaxis, ...], verbose=0)[0]
    print(f"\nSample inference for {sample_file.name}:")
    for label, probability in zip(MODEL_LABELS, sample_prediction):
        print(f"  {label}: {probability * 100.0:.2f}%")
    sample_prediction_plot = args.keras_model.parent / "no_wav_prediction.png"
    save_sample_prediction_plot(sample_prediction, sample_prediction_plot)

    tflite_model = convert_to_tflite(model, train_spectrogram_ds, args.tflite_model)
    inspect_tflite_model(tflite_model, args.keras_model.parent / "yes_no_model_quant_metadata.txt")

    print("\nArtifacts written:")
    print(f"  Keras model: {args.keras_model}")
    print(f"  Quantized TFLite model: {args.tflite_model}")
    print(f"  Training history plot: {history_plot}")
    print(f"  Confusion matrix plot: {confusion_plot}")
    print(f"  Sample prediction plot: {sample_prediction_plot}")
    print(f"  Demo sample: {sample_file}")


if __name__ == "__main__":
    main()
