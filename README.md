# 🔥 Semantic Embeddings Unleashed: A PyTorch Word2Vec CBOW  🔥

## Overview
This project implements a Continuous Bag of Words (CBOW) model using Word2Vec with PyTorch to learn word embeddings from a tokenized corpus. The model predicts a target word based on the surrounding context words and evaluates the learned embeddings using cosine similarity.

## Dataset
- The dataset is generated using a WordPiece tokenizer, which processes the input text into subword tokens.
- Context-target pairs are created using a window size of 2, ensuring that each target word is associated with surrounding context words.
- The dataset is split into 80% training and 20% validation.

## Model Architecture
The CBOW model consists of:
- An **embedding layer** with a size of 75 to learn word representations.
- A **dropout layer (0.5 probability)** for regularization.
- A **linear layer** that predicts the target word from the averaged context embeddings.

## Training
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Epochs:** 10
- **Batch Size:** 32
- Training involves feeding context word indices into the model and updating weights based on the loss.
- The model is saved at its best validation performance as `word2vec_best.pth`.
- Training and validation loss graphs are plotted for performance analysis.

## Plot
![plot](https://github.com/user-attachments/assets/d6978d3e-c832-4dc6-92a1-8c7c7b70d109)


## Cosine Similarity Evaluation
After training, the model evaluates the similarity between word embeddings:
- **Similarity between "feel" and "feeling":** Computed using cosine similarity.
- **Similarity between "feel" and "long":** Computed to compare semantic relationships.

These similarity results demonstrate the model's capability to capture semantic relationships between words.

## Dependencies
- Python
- PyTorch
- NumPy
- Matplotlib

## Usage
1. **Prepare the Corpus:** Store the text corpus in `corpus.txt`.
2. **Load the Vocabulary:** Ensure `vocabulary.txt` contains the vocabulary for tokenization.
3. **Preprocess Data:** The `Word2VecDataset` class tokenizes text and creates context-target pairs.
4. **Train the Model:** Run the training script to learn embeddings.
5. **Evaluate Word Embeddings:** Compute cosine similarity between word pairs to assess semantic relationships.
6. **Visualize Training Loss:** Loss plots are generated and saved as `training_validation_loss_w2v.png`.


