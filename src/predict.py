# src/predict.py

import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt

# Import necessary functions/classes from other src modules
try:
    from .feature_extraction import get_feature_extractor, encode_image
    from .model import create_caption_model
except ImportError:
    # Fallback for running script directly
    print("Using absolute imports assuming 'src' is in the Python path or relative to execution.")
    from feature_extraction import get_feature_extractor, encode_image
    from model import create_caption_model

# --- Prediction Functions ---
# You can either define greedySearch/beamSearch here OR import them if defined elsewhere

def greedy_search_predict(model, photo_encoding, wordtoix, ixtoword, max_length):
    """Generates a caption using greedy search."""
    start_token = 'startseq'
    end_token = 'endseq'
    in_text = start_token
    for _ in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post') # Crucial: padding='post'
        # Model expects list/tuple: [image_features, text_sequence]
        yhat = model.predict([photo_encoding, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = ixtoword.get(yhat_index, None) # Use get to handle potential index errors
        if word is None:
            break
        in_text += ' ' + word
        if word == end_token:
            break
    # Clean up final sequence
    final_words = in_text.split()
    final_words = [w for w in final_words if w not in [start_token, end_token]]
    return ' '.join(final_words)

# Define beam_search_decoder here if you want to use it as well
# def beam_search_decoder(model, photo_encoding, wordtoix, ixtoword, max_length, beam_width=3):
#     # ... (Paste the beam search implementation here) ...
#     pass


def main(args):
    """Loads model and generates caption for a single image."""

    print("--- Loading Vocabulary Mappings & Max Length ---")
    try:
        with open(args.vocab_path, 'rb') as f:
            mappings = pickle.load(f)
            wordtoix = mappings['wordtoix']
            ixtoword = mappings['ixtoword']
            vocab_size = mappings['vocab_size']
        with open(args.max_length_path, 'rb') as f:
            max_length = pickle.load(f)
        print(f"Loaded vocab (size={vocab_size}), mappings, and max_length ({max_length}).")
    except FileNotFoundError:
        print(f"ERROR: Vocab or max_length file not found. Check paths:")
        print(f"  Vocab path: {args.vocab_path}")
        print(f"  Max length path: {args.max_length_path}")
        return
    except Exception as e:
        print(f"Error loading vocab/max_length: {e}")
        return

    print("\n--- Loading Feature Extractor ---")
    feature_extractor = get_feature_extractor(weights=None) # Load architecture only

    print("\n--- Defining Captioning Model ---")
    caption_model = create_caption_model(vocab_size, max_length, args.embedding_dim,
                                         args.feature_dim, args.lstm_units)

    print("\n--- Loading Trained Weights ---")
    if not os.path.exists(args.weights_path):
        print(f"ERROR: Model weights file not found at {args.weights_path}")
        return
    try:
        caption_model.load_weights(args.weights_path)
        print(f"Loaded weights from {args.weights_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return


    print(f"\n--- Encoding Input Image: {args.image_path} ---")
    image_encoding = encode_image(args.image_path, feature_extractor)

    if image_encoding is None:
        print("ERROR: Could not encode the input image.")
        return

    # Reshape encoding for the captioning model's input layer
    image_encoding_reshaped = image_encoding.reshape((1, args.feature_dim))

    print("\n--- Generating Caption (Greedy Search) ---")
    start_pred_time = time.time()
    generated_caption = greedy_search_predict(caption_model, image_encoding_reshaped,
                                            wordtoix, ixtoword, max_length)
    end_pred_time = time.time()
    print(f"Caption generated in {end_pred_time - start_pred_time:.2f} seconds.")

    # --- Display Results ---
    print("\n--- Result ---")
    print(f"Generated Caption: {generated_caption}")

    if not args.no_display:
        print("\nDisplaying Image...")
        try:
            img_display = Image.open(args.image_path)
            plt.imshow(img_display)
            plt.title(f"Generated: {generated_caption}")
            plt.axis('off')
            plt.show()
        except FileNotFoundError:
            print(f"Error: Could not find image file at {args.image_path} to display.")
        except Exception as e:
            print(f"An error occurred displaying the image: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate caption for a single image.')

    # Required Arguments
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the trained model weights file (.weights.h5).')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary mappings pickle file (vocab_mappings.pkl).')
    parser.add_argument('--max_length_path', type=str, required=True, help='Path to the max length pickle file (max_length.pkl).')

    # Optional Model Hyperparameters (must match the trained model)
    parser.add_argument('--feature_dim', type=int, default=2048, help='Dimension of image features.')
    parser.add_argument('--embedding_dim', type=int, default=200, help='Dimension of word embeddings.')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of units in LSTM layer.')

    # Optional Prediction Arguments
    # parser.add_argument('--beam_width', type=int, default=3, help='Beam width if using Beam Search.') # Add if using beam search
    parser.add_argument('--no_display', action='store_true', help='Flag to suppress displaying the image.')

    args = parser.parse_args()

    main(args)