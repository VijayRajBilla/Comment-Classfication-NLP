# Comment-Classfication-NLP
Building a model using Data Extraction and NLP Techniques

# Toxic Comment Classification with Bidirectional GRU

This Python script implements a toxic comment classification model using Bidirectional Gated Recurrent Unit (GRU) layers. The model is trained on a dataset consisting of comments labeled with various toxicity categories, including "toxic," "severe_toxic," "obscene," "threat," "insult," and "identity_hate." The classification is performed using the Keras library with GloVe pre-trained word embeddings.

## Data Preprocessing

- The training data is loaded from 'train.csv,' and the test data is loaded from 'test.csv.'
- Missing values are handled by filling them with the string "fillna."
- Tokenization and padding are applied to convert text data into numerical sequences suitable for input to the neural network.

## Model Architecture

- The model architecture consists of an Embedding layer initialized with pre-trained GloVe word embeddings.
- Bidirectional GRU layers capture contextual information from both directions in the sequence.
- Global Average Pooling and Global Max Pooling layers are used to aggregate information from the sequence.
- The final dense layer with a sigmoid activation function outputs probabilities for each toxicity class.

## Training

- The model is trained using the binary cross-entropy loss function and the Adam optimizer.
- Model training is monitored using the accuracy metric.
- Checkpoints are saved during training based on the validation accuracy, and early stopping is applied to prevent overfitting.

## Prediction

- The trained model is used to predict toxicity probabilities for the comments in the test dataset.

## Dependencies

- Python 3.x
- Libraries: numpy, pandas, matplotlib, keras

## Usage

1. **Data Preparation:**
   - Place the training data in 'train.csv' and the test data in 'test.csv.'
   - Download the GloVe word embeddings file ('glove.6B.300d.txt') and update the 'embedding_file' variable accordingly.

2. **Training:**
   - Run the script to train the model. Checkpoints will be saved, and training progress will be displayed.

3. **Prediction:**
   - The trained model will be used to predict toxicity probabilities for the test data.

