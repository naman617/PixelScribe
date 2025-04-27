# src/feature_extraction.py

import numpy as np
import tensorflow as tf # Needed for Model, Input, etc. even if layers aren't used directly here sometimes
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image_proc
import os

# --- Image Preprocessing ---

def preprocess(image_path):
    """
    Loads and preprocesses an image file for InceptionV3.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image array ready for InceptionV3,
                       or None if file not found/load error.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    try:
        # Load image, ensuring target size for InceptionV3
        img = keras_image_proc.load_img(image_path, target_size=(299, 299))
        # Convert PIL image to numpy array
        x = keras_image_proc.img_to_array(img)
        # Add batch dimension
        x = np.expand_dims(x, axis=0)
        # Preprocess using InceptionV3's specific function
        x = preprocess_input(x)
        return x
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# --- Feature Extractor Model ---

def get_feature_extractor(weights='imagenet'):
    """
    Loads the pre-trained InceptionV3 model and modifies it
    to output features from the second-to-last layer.

    Args:
        weights (str): Weight initialization ('imagenet' or None).

    Returns:
        keras.Model: The feature extraction model.
    """
    # Load the base InceptionV3 model
    base_model = InceptionV3(weights=weights)
    # Create a new model that outputs the features from the layer before the final classification layer
    feature_extractor_model = Model(inputs=base_model.input,
                                    outputs=base_model.layers[-2].output)
    return feature_extractor_model

# --- Encoding Function ---

def encode_image(image_path, feature_extractor):
    """
    Encodes a single image file into a feature vector using the provided extractor.

    Args:
        image_path (str): Path to the image file.
        feature_extractor (keras.Model): The pre-trained feature extraction model (e.g., modified InceptionV3).

    Returns:
        numpy.ndarray: The feature vector (shape (2048,)), or None if preprocessing/prediction fails.
    """
    image_processed = preprocess(image_path)
    if image_processed is None: # Handle preprocessing errors
        return None

    try:
        # Get the feature vector
        fea_vec = feature_extractor.predict(image_processed, verbose=0)
        # Reshape from (1, 2048) to (2048,)
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
        return fea_vec
    except Exception as e:
        print(f"Error predicting features for {image_path}: {e}")
        return None

# --- Optional: Batch Encoding Function (useful for generating pkl files) ---

def encode_images_batch(image_paths, feature_extractor):
    """
    Encodes a batch of image files and returns features in a dictionary.

    Args:
        image_paths (list): List of full paths to image files.
        feature_extractor (keras.Model): The feature extraction model.

    Returns:
        dict: Dictionary mapping image filename (basename) to feature vector.
    """
    from tqdm import tqdm # Import tqdm locally for optional use
    import os

    encoding_dict = {}
    print(f"Encoding {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Encoding Images"):
        feature = encode_image(img_path, feature_extractor)
        if feature is not None:
            # Use basename (e.g., 'image.jpg') as key, consistent with pkl saving
            image_filename = os.path.basename(img_path)
            encoding_dict[image_filename] = feature
    print(f"Successfully encoded {len(encoding_dict)} images.")
    return encoding_dict