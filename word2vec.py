import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from task1 import WordPieceTokenizer  # Reuse your tokenizer from Task 1

# Custom Dataset for Word2Vec (CBOW)
class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, window_size=2):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.data_pairs = []

        tokenized_corpus = []
        for word in self.tokenizer.clean_text(corpus):  
            tokenized_corpus.extend(self.tokenizer.encode_single_word(word))  # Use subwords!


        # Generate CBOW context-target pairs
        for center_idx in range(len(tokenized_corpus)):
            context = []
            for offset in range(-window_size, window_size + 1):
                if offset != 0 and 0 <= center_idx + offset < len(tokenized_corpus):
                    context.append(tokenized_corpus[center_idx + offset])
            target = tokenized_corpus[center_idx]
            if len(context) == 2 * window_size:
                self.data_pairs.append((context, target))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        context, target = self.data_pairs[idx]
        context_ids = [self.tokenizer.vocab.index(sub) if sub in self.tokenizer.vocab else self.tokenizer.vocab.index("[UNK]") for sub in context]
        target_id = self.tokenizer.vocab.index(target) if target in self.tokenizer.vocab else self.tokenizer.vocab.index("[UNK]")
        return torch.tensor(context_ids), torch.tensor(target_id)


# CBOW Word2Vec Model
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout_rate=0.5):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        context_embeddings = self.embeddings(context_words)  # [batch_size, context_size, embedding_dim]
        context_mean = context_embeddings.mean(dim=1)  # [batch_size, embedding_dim]
        context_mean = self.dropout(context_mean)  # Apply dropout
        output = self.linear(context_mean)  # [batch_size, vocab_size]
        return output


def train(model, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for context, target in train_loader:
            optimizer.zero_grad()
            predictions = model(context)  # [batch_size, vocab_size]
            loss = loss_function(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                predictions = model(context)
                loss = loss_function(predictions, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "word2vec_best.pth")

    # Plot Training and Validation Loss
    plt.plot(range(1, epochs + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, marker="o", linestyle="dashed", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss vs. Epochs")
    plt.legend()
    plt.savefig("training_validation_loss_w2v.png")
    plt.show()

    print("Training complete. Best model saved as 'word2vec_best.pth'.")
 

# Compute Cosine Similarity between Word Embeddings
def compute_cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1 / torch.norm(embedding1, dim=-1, keepdim=True)
    embedding2 = embedding2 / torch.norm(embedding2, dim=-1, keepdim=True)
    return torch.dot(embedding1, embedding2).item()


if __name__ == "__main__":
    # Load corpus
    with open("corpus.txt", "r") as f:
        corpus = f.read()

    # Initialize tokenizer and load vocabulary
    tokenizer = WordPieceTokenizer()
    tokenizer.vocab = [line.strip() for line in open("vocabulary.txt", "r")]

    
    # Prepare dataset
    dataset = Word2VecDataset(corpus, tokenizer)
    vocab_size = len(tokenizer.vocab)

    # Split into train and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize model
    model = Word2VecModel(vocab_size=vocab_size, embedding_dim=75)

    # Train model
    train(model, train_dataset, val_dataset, epochs=10, batch_size=32)

    # Load the best model for cosine similarity evaluation
    model.load_state_dict(torch.load("word2vec_best.pth"))
    model.eval()

    # Identify two triplets using cosine similarity
    with torch.no_grad():
        word1, word2, word3 = "feel", "feeling", "long"

        if word1 in tokenizer.vocab and word2 in tokenizer.vocab and word3 in tokenizer.vocab:
            embedding1 = model.embeddings(torch.tensor([tokenizer.vocab.index(word1)])).squeeze(0)
            embedding2 = model.embeddings(torch.tensor([tokenizer.vocab.index(word2)])).squeeze(0)
            embedding3 = model.embeddings(torch.tensor([tokenizer.vocab.index(word3)])).squeeze(0)

            similarity12 = compute_cosine_similarity(embedding1, embedding2)
            similarity13 = compute_cosine_similarity(embedding1, embedding3)

            print(f"Similarity between {word1} and {word2}: {similarity12}")
            print(f"Similarity between {word1} and {word3}: {similarity13}")
