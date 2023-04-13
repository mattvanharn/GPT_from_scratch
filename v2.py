import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32  # How many independent sequences we will process in parallel
block_size = 8  # The maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cude' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
# --------------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return "".join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading


def get_batch(split):
    '''Generate a small batch of data of inputs x and targets y'''
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        ''' idk and targets are both (B, T) tensor of integers '''
        B, T = idx.shape

        # ((B, T, C) Batch = 4, Time = 8, Channels = vocab_size = 65)
        tok_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(
            torch.arange(T, device=device))  # (T, C)
        # x holds the token identities and the positions at which they occur
        x = tok_embed + pos_embed  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        ''' idx is (B, T) array of indices in the current context'''
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # Become (B, C)
            # Apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and valid sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Iteration {iter} Train loss: {losses['train']:.3f} Val loss: {losses['val']:.3f}")

    # Sample a bunch of data
    xb, yb = get_batch('train')

    # Compute the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
