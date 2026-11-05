# The models use tf.GradientTape / tf.custom_gradient, so Keras must run on the TensorFlow backend in float32.
# A ~/.keras/keras.json with "backend": "jax" / "floatx": "float64" would otherwise break training
# ("'Variable' object has no attribute '_id'").
import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"


def tf_keras():
    """
    Returns Keras running on TensorFlow in float32. If this Python session (e.g. a Spyder console) already
    imported Keras with another backend, Keras is reloaded on TensorFlow; Keras objects created earlier in
    that session cannot be used afterwards.
    """
    import keras
    if keras.backend.backend() != "tensorflow":
        previous = keras.backend.backend()
        keras.config.set_backend("tensorflow")
        keras = sys.modules["keras"]
        print(f"[review29oct_fixed] Keras was loaded with the '{previous}' backend in this session; "
              f"reloaded it on 'tensorflow'.")
    keras.backend.set_floatx("float32")
    return keras
