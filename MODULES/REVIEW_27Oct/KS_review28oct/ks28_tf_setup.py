# -*- coding: utf-8 -*-
"""
ks28_tf_setup.py -- Make sure Keras runs on the TensorFlow backend.

Keras 3 can run on JAX, PyTorch or TensorFlow (chosen by the KERAS_BACKEND environment
variable or ~/.keras/keras.json). This code trains with tf.GradientTape, which only sees
TensorFlow variables, so the backend MUST be "tensorflow". With the JAX backend the error
is:  AttributeError: 'Variable' object has no attribute '_id'.

This module is imported by every TF module of KS_review28oct BEFORE tensorflow/keras.
If Keras was already imported with another backend in the same Python session (e.g. a
Spyder console where something else ran first), the backend cannot be changed any more:
restart the kernel and run again.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf  # noqa: E402


def _check_backend():
    try:
        backend = tf.keras.backend.backend()
    except Exception:          # very old Keras: always TensorFlow
        return
    if backend != "tensorflow":
        raise RuntimeError(
            f"Keras is running on the '{backend}' backend, but KS_review28oct needs 'tensorflow'.\n"
            "Keras was imported with another backend earlier in this Python session.\n"
            "Fix: restart the kernel (Spyder: Consoles > Restart kernel) and run again.\n"
            "Permanent fix: set \"backend\": \"tensorflow\" in %USERPROFILE%\\.keras\\keras.json "
            "or define the environment variable KERAS_BACKEND=tensorflow.")


_check_backend()

# The model works in float32 (the eigensolver is float64 internally). keras.json may set
# "floatx": "float64", which would make the encoder output float64 and break the graph.
try:
    tf.keras.backend.set_floatx("float32")
except Exception:
    pass
