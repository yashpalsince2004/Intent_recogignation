import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("intent_model.h5")

# Set up the converter with the required flags
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # Standard ops
    tf.lite.OpsSet.SELECT_TF_OPS          # Enable TF ops not natively supported in TFLite
]
converter._experimental_lower_tensor_list_ops = False  # Prevent TensorList lowering

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("intent_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ intent_model.tflite converted and saved successfully!")
