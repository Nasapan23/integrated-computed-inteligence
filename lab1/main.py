import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

x = np.linspace(0.0, 2.0 * np.pi, 400)
noise = np.random.normal(0.0, 0.05, size=x.shape)
y = np.sin(x) + noise

perm = np.random.permutation(len(x))
x = x[perm]
y = y[perm]

n = len(x)
train_end = int(0.2 * n)
val_end = int(0.8 * n)
x_train, y_train = x[:train_end], y[:train_end]
x_val, y_val = x[train_end:val_end], y[train_end:val_end]
x_test, y_test = x[val_end:], y[val_end:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, verbose=0)
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)

x_plot = np.linspace(0.0, 2.0 * np.pi, 400)
y_true = np.sin(x_plot)
y_pred = model.predict(x_plot, verbose=0).squeeze()

plt.figure(figsize=(8, 4))
plt.plot(x_plot, y_true, label='sin(x)')
plt.plot(x_plot, y_pred, label='model')
plt.scatter(x_train, y_train, s=10, alpha=0.6, label='train')
plt.scatter(x_val, y_val, s=10, alpha=0.6, label='validate')
plt.scatter(x_test, y_test, s=10, alpha=0.6, label='test')
plt.legend()
plt.tight_layout()
plt.show()
