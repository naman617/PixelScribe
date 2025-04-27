# src/generator.py

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os 

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch, vocab_size):
    """
    Generates batches of data (image features, text sequences, target words) for model training.

    Args:
        descriptions (dict): Dictionary mapping image IDs to lists of 'startseq ... endseq' captions.
        photos (dict): Dictionary mapping image filenames (e.g., 'img.jpg') to feature vectors.
        wordtoix (dict): Dictionary mapping words to indices.
        max_length (int): Maximum length for padding sequences.
        num_photos_per_batch (int): Number of unique photos to process before yielding a batch.
        vocab_size (int): Total size of the vocabulary (for one-hot encoding).

    Yields:
        tuple: A tuple containing ([batch_image_features, batch_input_sequences], batch_output_words),
               where batches contain data from 'num_photos_per_batch' images.
    """
    X1, X2, y = list(), list(), list()
    n = 0
    # Loop indefinitely
    while True: # Changed from 'while 1' for clarity
        # Shuffle keys each epoch? Optional, but can help training.
        # items = list(descriptions.items())
        # random.shuffle(items)
        # for key, desc_list in items:
        # Simpler loop without shuffling for now:
        for key, desc_list in descriptions.items():
            n += 1
            # Construct photo key (ensure .jpg extension)
            photo_key = key
            if not key.endswith('.jpg'):
                 photo_key = key + '.jpg'

            # Retrieve the photo feature, skip if missing
            if photo_key not in photos:
                # print(f"Warning: Photo key '{photo_key}' not found in photos dictionary. Skipping in generator.")
                continue # Skip this image if features aren't loaded
            photo = photos[photo_key]

            for desc in desc_list:
                # Encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # Split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # Split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # Pad input sequence (post-padding)
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # Encode output sequence (one-hot)
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # Store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            # Yield the batch data when enough photos processed
            if n == num_photos_per_batch:
                # Yield a tuple: (list_of_inputs, outputs)
                # Ensure inputs list/tuple matches model input structure
                yield ( (np.array(X1), np.array(X2)), np.array(y) )
                # Reset for next batch
                X1, X2, y = list(), list(), list()
                n = 0
        # It's possible the loop finishes without n reaching num_photos_per_batch exactly.
        # If you want to yield the remaining items as a potentially smaller final batch:
        # (Optional - depends if your training loop handles variable last batch size)
        # if X1: # If there's anything left
        #    yield ( (np.array(X1), np.array(X2)), np.array(y) )
        #    X1, X2, y = list(), list(), list()
        #    n = 0
        # Note: Keras fit usually expects the generator to loop forever, so the simple
        #       reset after hitting the batch size is standard.