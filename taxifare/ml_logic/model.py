import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(input_shape: tuple) -> Model:
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    print("✅ Model initialized")
    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae"]
    )

    print("✅ Model compiled")
    return model


def train_model(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size=256,
    patience=2,
    validation_data=None,
    validation_split=0.3
) -> Tuple[Model, dict]:

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=100,
        validation_data=validation_data,
        validation_split=None if validation_data else validation_split,
        callbacks=[es],
        verbose=0
    )

    print(
        f"✅ Model trained on {len(X)} rows "
        f"with min val MAE: {round(np.min(history.history['val_mae']), 2)}"
    )

    return model, history

