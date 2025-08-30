import numpy as np
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from build_model.build_tensor import *

# Build the neural network model
def create_fantasy_model(input_dim):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='silu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        layers.Dense(32, activation='silu'),
        layers.Dense(1, kernel_constraint=keras.constraints.MaxNorm(3.0))  # Single output for fantasy points
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model

# Create the model
model = create_fantasy_model(num_features)
print(f"Model created with {num_features} input features")
model.summary()

# Define callbacks
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
]

# Train the model
print("Starting training...")
history = model.fit(
    train_dataset,
    epochs=55,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
test_loss, test_mae, test_mse = model.evaluate(test_dataset, verbose=0)
print(f"\nTest Results:")
print(f"- Test Loss: {test_loss:.4f}")
print(f"- Test MAE: {test_mae:.4f}")
print(f"- Test RMSE: {np.sqrt(test_mse):.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Make predictions on test set
test_predictions = model.predict(test_dataset)

# Calculate additional metrics
from sklearn.metrics import r2_score
r2 = r2_score(y_test, test_predictions)
print(f"\nAdditional Metrics:")
print(f"- R² Score: {r2:.4f}")

# Save the model
model.save('fantasy_football_model.keras')
print("\nModel saved as 'fantasy_football_model.keras'")

print("\nModel training complete!")
print(f"You can now use this model to predict fantasy football performance.")
print(f"The model uses {len(feature_names)} features to predict fantasy points.")
