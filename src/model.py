# src/model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import plot_model 

def create_caption_model(vocab_size, max_length, embedding_dim=200, feature_dim=2048, lstm_units=256):
    """
    Defines the CNN-LSTM image captioning model architecture.

    Args:
        vocab_size (int): Size of the vocabulary (including padding).
        max_length (int): Maximum length of input caption sequences.
        embedding_dim (int): Dimension of the word embeddings (e.g., from GloVe).
        feature_dim (int): Dimension of the input image features (e.g., 2048 from InceptionV3).
        lstm_units (int): Number of units in the LSTM layer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    # --- Feature Extractor Input ---
    # Input shape is the size of the image feature vector
    inputs1 = Input(shape=(feature_dim,), name='image_features_input')
    # Dropout applied to image features
    fe1 = Dropout(0.5)(inputs1)
    # Dense layer to process image features (adjust dimensionality if needed)
    fe2 = Dense(lstm_units, activation='relu')(fe1) # Match LSTM units

    # --- Sequence Model Input ---
    # Input shape is the max length of sequences
    inputs2 = Input(shape=(max_length,), name='sequence_input')
    # Embedding layer (mask_zero=True is important for handling padding)
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    # Dropout applied to embeddings
    se2 = Dropout(0.5)(se1)
    # LSTM layer
    se3 = LSTM(lstm_units)(se2) # Using stateful=False (default)

    # --- Decoder (Merging Features and Sequence) ---
    # Merge the processed image features and LSTM output
    decoder1 = add([fe2, se3])
    # Dense layer after merging
    decoder2 = Dense(lstm_units, activation='relu')(decoder1)
    # Final output layer: Dense layer with softmax activation for probability distribution over vocab
    outputs = Dense(vocab_size, activation='softmax', name='output_word')(decoder2)

    # --- Create Model ---
    # Ties the inputs and outputs together
    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='caption_model')

    # --- Compile Model (Optional - can also be done in train.py) ---
    # It's often better to compile in the training script where optimizer details might vary,
    # but compiling here ensures the model is ready if used directly.
    # model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

if __name__ == '__main__':
    # Example usage and summary print if script is run directly
    print("Example Model Creation:")
    # Define some dummy values for demonstration
    example_vocab_size = 10000
    example_max_length = 34
    example_embedding_dim = 200
    example_feature_dim = 2048

    example_model = create_caption_model(example_vocab_size, example_max_length,
                                         example_embedding_dim, example_feature_dim)
    example_model.summary()

    # Optional: Save a plot of the model
    # try:
    #     plot_model(example_model, to_file='caption_model_plot.png', show_shapes=True)
    #     print("Saved model plot to caption_model_plot.png")
    # except ImportError:
    #     print("Could not plot model: pydot and graphviz might be required.")
    #     print("Install with: pip install pydot graphviz")