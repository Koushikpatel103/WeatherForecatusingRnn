from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN,
    Dense,
    Dropout
)


def build_rnn_model(input_shape):

    model = Sequential()

    model.add(
        SimpleRNN(
            units=64,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        SimpleRNN(
            units=64
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(25, activation="relu")
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model