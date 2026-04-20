"""
Fine-tuned LLM Evaluation

Clean Python script generated from the notebook for easier reading and execution.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Fine-tuned LLM evaluation workflow converted from notebook format."
        )
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=str(Path(__file__).with_name("the-verdict.txt")),
        help="Path to the input text file used by tokenization/training sections."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ==============================================================================
    # Section 1: Reading in a short story as text sample into Python.
    # ==============================================================================

    # ==============================================================================
    # Section 2: Step 1: Creating Tokens
    # ==============================================================================

    with open(args.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    import re

    text = "Hello, world. This, is a test."
    result = re.split(r'(\s)', text)

    print(result)

    result = re.split(r'([,.]|\s)', text)

    print(result)

    result = [item for item in result if item.strip()]
    print(result)

    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)

    # Strip whitespace from each item and then filter out any empty strings.
    result = [item for item in result if item.strip()]
    print(result)

    text = "Hello, world. Is this-- a test?"

    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])

    print(len(preprocessed))

    # ==============================================================================
    # Section 3: Step 2: Creating Token IDs
    # ==============================================================================

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)

    print(vocab_size)

    vocab = {token:integer for integer,token in enumerate(all_words)}

    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break

    class SimpleTokenizerV1:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i:s for s,i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            preprocessed = [
                item.strip() for item in preprocessed if item.strip()
            ]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return text

    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    tokenizer.decode(ids)

    text = "Hello, do you like tea?"
    print(tokenizer.encode(text))

    # ==============================================================================
    # Section 4: ADDING SPECIAL CONTEXT TOKENS
    # ==============================================================================

    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token:integer for integer,token in enumerate(all_tokens)}

    len(vocab.items())

    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    class SimpleTokenizerV2:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = { i:s for s,i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [
                item if item in self.str_to_int 
                else "<|unk|>" for item in preprocessed
            ]

            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
            return text

    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    print(text)

    tokenizer.encode(text)

    tokenizer.decode(tokenizer.encode(text))

    # ==============================================================================
    # Section 5: BYTE PAIR ENCODING (BPE)
    # ==============================================================================

    # Notebook shell command removed: ! pip3 install tiktoken

    import importlib
    import tiktoken

    print("tiktoken version:", importlib.metadata.version("tiktoken"))

    tokenizer = tiktoken.get_encoding("gpt2")

    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )

    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    print(integers)

    strings = tokenizer.decode(integers)

    print(strings)

    integers = tokenizer.encode("Akwirw ier")
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)

    import tiktoken

    # Initialize the encodings for GPT-2, GPT-3, and GPT-4
    encodings = {
        "gpt2": tiktoken.get_encoding("gpt2"),
        "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
        "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
    }

    # Get the vocabulary size for each encoding
    vocab_sizes = {model: encoding.n_vocab for model, encoding in encodings.items()}

    # Print the vocabulary sizes
    for model, size in vocab_sizes.items():
        print(f"The vocabulary size for {model.upper()} is: {size}")

    # ==============================================================================
    # Section 6: CREATING INPUT-TARGET PAIRS
    # ==============================================================================

    with open(args.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    enc_sample = enc_text[50:]

    context_size = 4 #length of the input
    #The context_size of 4 means that the model is trained to look at a sequence of 4 words (or tokens) 
    #to predict the next word in the sequence. 
    #The input x is the first 4 tokens [1, 2, 3, 4], and the target y is the next 4 tokens [2, 3, 4, 5]

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"x: {x}")
    print(f"y:      {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(context, "---->", desired)

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

    # ==============================================================================
    # Section 7: IMPLEMENTING A DATA LOADER
    # ==============================================================================

    from torch.utils.data import Dataset, DataLoader


    class GPTDatasetV1(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            # Tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]

    def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                             stride=128, shuffle=True, drop_last=True,
                             num_workers=0):

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader

    with open(args.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    import torch
    print("PyTorch version:", torch.__version__)
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

    # ==============================================================================
    # Section 8: CREATING TOKEN EMBEDDINGS
    # ==============================================================================

    input_ids = torch.tensor([2, 3, 5, 1])

    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    print(embedding_layer.weight)

    print(embedding_layer(torch.tensor([3])))

    print(embedding_layer(input_ids))

    # ==============================================================================
    # Section 9: POSITIONAL EMBEDDINGS (ENCODING WORD POSITIONS)
    # ==============================================================================

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    # ==============================================================================
    # Section 10: IMPLEMENTING A SIMPLIFIED ATTENTION MECHANISM
    # ==============================================================================

    import torch

    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )

    # Create 3D plot with vectors from origin to each point, using different colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a list of colors for the vectors
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Plot each vector with a different color and annotate with the corresponding word
    for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words, colors):
        # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
        ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
        ax.text(x, y, z, word, fontsize=10, color=color)

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits to keep arrows within the plot boundaries
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.title('3D Plot of Word Embeddings with Colored Vectors')
    plt.show()

    query = inputs[1]  # 2nd input token is the query

    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

    print(attn_scores_2)

    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)

    attn_weights_2_naive = softmax_naive(attn_scores_2)

    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    query = inputs[1] # 2nd input token is the query

    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i

    print(context_vec_2)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    inputs2 = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55], # step     (x^6)
       [0.4419, 0.6515, 0.5683]]
    )

    # Corresponding words
    words2 = ['Your', 'journey', 'starts', 'with', 'one', 'step', 'journey-context']

    # Extract x, y, z coordinates
    x_coords = inputs2[:, 0].numpy()
    y_coords = inputs2[:, 1].numpy()
    z_coords = inputs2[:, 2].numpy()

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point and annotate with corresponding word
    for x, y, z, word in zip(x_coords, y_coords, z_coords, words2):
        ax.scatter(x, y, z)
        ax.text(x, y, z, word, fontsize=10)

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('3D Plot of Word Embeddings')
    plt.show()

    # Create 3D plot with vectors from origin to each point, using different colors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a list of colors for the vectors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'r']

    # Plot each vector with a different color and annotate with the corresponding word
    for (x, y, z, word, color) in zip(x_coords, y_coords, z_coords, words2, colors):
        # Draw vector from origin to the point (x, y, z) with specified color and smaller arrow length ratio
        ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.05)
        ax.text(x, y, z, word, fontsize=10, color=color)

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits to keep arrows within the plot boundaries
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.title('3D Plot of Word Embeddings with Colored Vectors')
    plt.show()

    attn_scores = torch.empty(6, 6)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)

    print(attn_scores)

    attn_scores = inputs @ inputs.T
    print(attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)

    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))

    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)

    print("Previous 2nd context vector:", context_vec_2)

    # ==============================================================================
    # Section 11: IMPLEMENTING SELF ATTENTION WITH TRAINABLE WEIGHTS
    # ==============================================================================

    import torch

    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )

    # ==============================================================================
    # Section 12: A The second input element
    # ==============================================================================

    x_2 = inputs[1] #A
    d_in = inputs.shape[1] #B
    d_out = 2 #C

    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    print(W_query)

    print(W_key)

    print(W_value)

    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)

    keys = inputs @ W_key
    values = inputs @ W_value
    queries = inputs @ W_query
    print("keys.shape:", keys.shape)

    print("values.shape:", values.shape)

    print("queries.shape:", queries.shape)

    keys_2 = keys[1] #A
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)

    attn_scores_2 = query_2 @ keys.T # All attention scores for given query
    print(attn_scores_2)

    attn_scores = queries @ keys.T # omega
    print(attn_scores)

    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print(attn_weights_2)
    print(d_k)

    # ==============================================================================
    # Section 13: WHY DIVIDE BY SQRT (DIMENSION)
    # ==============================================================================

    import torch

    # Define the tensor
    tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

    # Apply softmax without scaling
    softmax_result = torch.softmax(tensor, dim=-1)
    print("Softmax without scaling:", softmax_result)

    # Multiply the tensor by 8 and then apply softmax
    scaled_tensor = tensor * 8
    softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
    print("Softmax after scaling (tensor * 8):", softmax_scaled_result)

    # ==============================================================================
    # Section 14: BUT WHY SQRT?
    # ==============================================================================

    import numpy as np

    # Function to compute variance before and after scaling
    def compute_variance(dim, num_trials=1000):
        dot_products = []
        scaled_dot_products = []

        # Generate multiple random vectors and compute dot products
        for _ in range(num_trials):
            q = np.random.randn(dim)
            k = np.random.randn(dim)

            # Compute dot product
            dot_product = np.dot(q, k)
            dot_products.append(dot_product)

            # Scale the dot product by sqrt(dim)
            scaled_dot_product = dot_product / np.sqrt(dim)
            scaled_dot_products.append(scaled_dot_product)

        # Calculate variance of the dot products
        variance_before_scaling = np.var(dot_products)
        variance_after_scaling = np.var(scaled_dot_products)

        return variance_before_scaling, variance_after_scaling

    # For dimension 5
    variance_before_5, variance_after_5 = compute_variance(5)
    print(f"Variance before scaling (dim=5): {variance_before_5}")
    print(f"Variance after scaling (dim=5): {variance_after_5}")

    # For dimension 20
    variance_before_100, variance_after_100 = compute_variance(100)
    print(f"Variance before scaling (dim=100): {variance_before_100}")
    print(f"Variance after scaling (dim=100): {variance_after_100}")


    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)

    # ==============================================================================
    # Section 15: IMPLEMENTING A COMPACT SELF ATTENTION PYTHON CLASS
    # ==============================================================================

    import torch.nn as nn

    class SelfAttention_v1(nn.Module):

        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            attn_scores = queries @ keys.T # omega
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    class SelfAttention_v2(nn.Module):

        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    # ==============================================================================
    # Section 16: HIDING FUTURE WORDS WITH CAUSAL ATTENTION
    # ==============================================================================

    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )

    queries = sa_v2.W_query(inputs) #A
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
    print(attn_weights)

    torch.ones(context_length, context_length)

    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    masked_simple = attn_weights*mask_simple
    print(masked_simple)

    row_sums = masked_simple.sum(dim=1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)

    print(attn_scores)

    torch.triu(torch.ones(context_length, context_length))

    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    print(mask)

    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)

    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
    print(attn_weights)

    # ==============================================================================
    # Section 17: MASKING ADDITIONAL ATTENTION WEIGHTS WITH DROPOUT
    # ==============================================================================

    example = torch.ones(6, 6) #B
    print(example)

    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5) #A
    example = torch.ones(6, 6) #B
    print(dropout(example))

    torch.manual_seed(123)
    print(dropout(attn_weights))

    # ==============================================================================
    # Section 18: IMPLEMENTING A COMPACT CAUSAL ATTENTION CLASS
    # ==============================================================================

    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) 

    class CausalAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length,
                     dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout) # New
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

        def forward(self, x):
            b, num_tokens, d_in = x.shape # New batch dimension b
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
            attn_scores.masked_fill_(  # New, _ ops are in-place
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )
            attn_weights = self.dropout(attn_weights) # New

            context_vec = attn_weights @ values
            return context_vec

    print(d_in)

    print(d_out)

    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vecs.shape:", context_vecs.shape)

    print(context_vecs)

    # ==============================================================================
    # Section 19: EXTENDING SINGLE HEAD ATTENTION TO MULTI-HEAD ATTENTION
    # ==============================================================================

    class MultiHeadAttentionWrapper(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            self.heads = nn.ModuleList(
                [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
                 for _ in range(num_heads)]
            )

        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)

    inputs = torch.tensor(
      [[0.43, 0.15, 0.89], # Your     (x^1)
       [0.55, 0.87, 0.66], # journey  (x^2)
       [0.57, 0.85, 0.64], # starts   (x^3)
       [0.22, 0.58, 0.33], # with     (x^4)
       [0.77, 0.25, 0.10], # one      (x^5)
       [0.05, 0.80, 0.55]] # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) 

    torch.manual_seed(123)
    context_length = batch.shape[1] # This is the number of tokens = 6
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    # ==============================================================================
    # Section 20: IMPLEMENTING MULTI-HEAD ATTENTION WITH WEIGHT SPLITS
    # ==============================================================================

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert (d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length),
                           diagonal=1)
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape

            keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            # We implicitly split the matrix by adding a `num_heads` dimension
            # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            # Compute scaled dot-product attention (aka self-attention) with a causal mask
            attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Shape: (b, num_tokens, num_heads, head_dim)
            context_vec = (attn_weights @ values).transpose(1, 2) 

            # Combine heads, where self.d_out = self.num_heads * self.head_dim
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec) # optional projection

            return context_vec

    # ==============================================================================
    # Section 21: DETAILED EXPLANATION OF THE MULTI-HEAD ATTENTION CLASS
    # ==============================================================================

    torch.manual_seed(123)

    # Define the tensor with 3 rows and 6 columns
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
         [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
         [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
    )

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) 

    batch_size, context_length, d_in = batch.shape
    d_out = 6
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    # ==============================================================================
    # Section 22: IMPLEMENTING A GPT MODEL FROM SCRATCH TO GENERATE TEXT
    # ==============================================================================

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    # ==============================================================================
    # Section 23: GPT ARCHITECTURE PART 1: DUMMY GPT MODEL CLASS
    # ==============================================================================

    import torch
    import torch.nn as nn


    class DummyGPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # Use a placeholder for TransformerBlock
            self.trf_blocks = nn.Sequential(
                *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            # Use a placeholder for LayerNorm
            self.final_norm = DummyLayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits


    class DummyTransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # A simple placeholder

        def forward(self, x):
            # This block does nothing and just returns its input.
            return x


    class DummyLayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()
            # The parameters here are just to mimic the LayerNorm interface.

        def forward(self, x):
            # This layer does nothing and just returns its input.
            return x

    # ==============================================================================
    # Section 24: STEP 1: TOKENIZATION
    # ==============================================================================

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # ==============================================================================
    # Section 25: STEP 2: CREATE AN INSTANCE OF DUMMYGPTMODEL
    # ==============================================================================

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

    # ==============================================================================
    # Section 26: GPT ARCHITECTURE PART 2: LAYER NORMALIZATION
    # ==============================================================================

    # ==============================================================================
    # Section 27: Explanation with a simple example
    # ==============================================================================

    torch.manual_seed(123)
    batch_example = torch.randn(2, 5) #A
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    print(batch_example)

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    # ==============================================================================
    # Section 28: GPT ARCHITECTURE PART 3: FEEDFORWARD NEURAL NETWORK WITH GELU ACTIVATION
    # ==============================================================================

    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3))
            ))

    import matplotlib.pyplot as plt

    gelu, relu = GELU(), nn.ReLU()

    # Some sample data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
                GELU(), ## Activation
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
            )

        def forward(self, x):
            return self.layers(x)

    print(GPT_CONFIG_124M["emb_dim"])

    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768) #A
    out = ffn(x)
    print(out.shape)

    # ==============================================================================
    # Section 29: GPT ARCHITECTURE PART 4: SHORTCUT CONNECTIONS
    # ==============================================================================

    class ExampleDeepNeuralNetwork(nn.Module):
        def __init__(self, layer_sizes, use_shortcut):
            super().__init__()
            self.use_shortcut = use_shortcut
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
            ])

        def forward(self, x):
            for layer in self.layers:
                # Compute the output of the current layer
                layer_output = layer(x)
                # Check if shortcut can be applied
                if self.use_shortcut and x.shape == layer_output.shape:
                    x = x + layer_output
                else:
                    x = layer_output
            return x

    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123) # specify random seed for the initial weights for reproducibility
    model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
    )

    def print_gradients(model, x):
        # Forward pass
        output = model(x)
        target = torch.tensor([[0.]])

        # Calculate loss based on how close the target
        # and output are
        loss = nn.MSELoss()
        loss = loss(output, target)

        # Backward pass to calculate the gradients
        loss.backward()

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Print the mean absolute gradient of the weights
                print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)

    # ==============================================================================
    # Section 30: GPT ARCHITECTURE PART 5: CODING ATTENTION AND LINEAR LAYERS IN A TRANSFORMER BLOCK
    # ==============================================================================

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    # ==============================================================================
    # Section 31: THE BUILDING BLOCKS: LAYER NORMALIZATION, GELU AND FEED-FORWARD NEURAL NETWORK
    # ==============================================================================

    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3))
            ))


    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
                GELU(), ## Activation
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
            )

        def forward(self, x):
            return self.layers(x)

    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"], 
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            # 2*4*768
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x
            # 2*4*768

    torch.manual_seed(123)
    x = torch.rand(2, 4, 768) #A
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # ==============================================================================
    # Section 32: GPT ARCHITECTURE PART 6: ENTIRE GPT MODEL ARCHITECTURE IMPLEMENTATION
    # ==============================================================================

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    import torch
    import torch.nn as nn


    class DummyGPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # Use a placeholder for TransformerBlock
            self.trf_blocks = nn.Sequential(
                *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            # Use a placeholder for LayerNorm
            self.final_norm = DummyLayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits


    class DummyTransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # A simple placeholder

        def forward(self, x):
            # This block does nothing and just returns its input.
            return x


    class DummyLayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()
            # The parameters here are just to mimic the LayerNorm interface.

        def forward(self, x):
            # This layer does nothing and just returns its input.
            return x

    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                (x + 0.044715 * torch.pow(x, 3))
            ))


    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
                GELU(), ## Activation
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
            )

        def forward(self, x):
            return self.layers(x)

    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"], 
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            # 2*4*768
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x
            # 2*4*768

    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    total_size_bytes = total_params * 4 #A
    total_size_mb = total_size_bytes / (1024 * 1024) #B
    print(f"Total size of the model: {total_size_mb:.2f} MB")

    # ==============================================================================
    # Section 33: GPT ARCHITECTURE PART 7: GENERATING TEXT FROM OUTPUT TOKENS
    # ==============================================================================

    def generate_text_simple(model, idx, max_new_tokens, context_size):
        # idx is (batch, n_tokens) array of indices in the current context

        ###Input batch:
     ###tensor([[6109, 3626, 6100,  345],
            ##[6109, 1110, 6622,  257]])

        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = model(idx_cond) ### batch, n_tokens, vocab_size

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]  

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() #A
    #model = GPTModel(GPT_CONFIG_124M)
    out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

    # ==============================================================================
    # Section 34: EVALUATING GENERATIVE TEXT MODELS
    # ==============================================================================

    # ==============================================================================
    # Section 35: GPT Model class we coded earlier
    # ==============================================================================

    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    # ==============================================================================
    # Section 36: Using GPT to generate text
    # ==============================================================================

    import torch

    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval();  # Disable dropout during inference

    import tiktoken

    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
        return encoded_tensor

    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0) # remove batch dimension
        return tokenizer.decode(flat.tolist())

    start_context = "Every effort moves you"



    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # ==============================================================================
    # Section 37: Calculating the text generation loss: cross-entropy and perplexity
    # ==============================================================================

    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # ==============================================================================
    # Section 38: Cross-entropy loss
    # ==============================================================================

    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    # Compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    # Calculate the average probability for each token
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    # Logits have shape (batch_size, num_tokens, vocab_size)
    print("Logits shape:", logits.shape)

    # Targets have shape (batch_size, num_tokens)
    print("Targets shape:", targets.shape)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)

    # ==============================================================================
    # Section 39: Perplexity
    # ==============================================================================

    perplexity = torch.exp(loss)
    print(perplexity)

    # ==============================================================================
    # Section 40: Calculating the training and validation set losses
    # ==============================================================================

    import os
    import urllib.request

    file_path = args.text_file
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # First 100 characters
    print(text_data[:99])

    # Last 100 characters
    print(text_data[-99:])

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    # ==============================================================================
    # Section 41: Implementing the DataLoader:
    # ==============================================================================

    from torch.utils.data import Dataset, DataLoader


    class GPTDatasetV1(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            # Tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]


    def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                             stride=128, shuffle=True, drop_last=True,
                             num_workers=0):

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader

    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]


    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Sanity check

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "decrease the `training_ratio`")

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    print(len(train_loader))
    print(len(val_loader))

    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    # ==============================================================================
    # Section 42: Here is the GPT Model class we coded earlier. We will need this
    # ==============================================================================

    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval();  # Disable dropout during inference

    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss


    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note:
    # Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # However, the resulting loss values may be slightly different.

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    #
    # print(f"Using {device} device.")


    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes


    torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    # ==============================================================================
    # Section 43: TRAINING LOOP FOR THE LLM
    # ==============================================================================

    def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0: 
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

        return train_losses, val_losses, track_tokens_seen

    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

    def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()

    # Note:
    # Uncomment the following code to calculate the execution time
    import time
    start_time = time.time()

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)›
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Note:
    # Uncomment the following code to show the execution time
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator


    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig("loss-plot.pdf")
        plt.show()

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # ==============================================================================
    # Section 44: DECODING STRATEGIES TO CONTROL RANDOMNESS
    # ==============================================================================

    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # ==============================================================================
    # Section 45: DECODING STRATEGY 1: TEMPERATURE SCALING
    # ==============================================================================

    vocab = { 
        "closer": 0,
        "every": 1, 
        "effort": 2, 
        "forward": 3,
        "inches": 4,
        "moves": 5, 
        "pizza": 6,
        "toward": 7,
        "you": 8,
    } 

    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    next_token_logits2 = next_token_logits/0.1

    next_token_logits3 = next_token_logits/5

    probas = torch.softmax(next_token_logits2, dim=0)

    print(probas)


    probas = torch.softmax(next_token_logits3, dim=0)

    print(probas)

    probas = torch.softmax(next_token_logits, dim=0)

    print(probas)

    next_token_id = torch.argmax(probas).item()

    print(next_token_id)

    print(inverse_vocab[next_token_id])

    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    def print_sampled_tokens(probas):
        torch.manual_seed(123) # Manual seed for reproducibility
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample))
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")

    print_sampled_tokens(probas)

    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)

    # Temperature values
    temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

    # Calculate scaled probabilities
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

    ##Multinomial

    # Plotting
    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig("temperature-plot.pdf")
    plt.show()

    # ==============================================================================
    # Section 46: DECODING STRATEGY 2: Top-k sampling
    # ==============================================================================

    next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)

    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")), 
        other=next_token_logits
    )

    print(new_logits)

    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

    # ==============================================================================
    # Section 47: Merge Temperature Scaling and Top-k sampling
    # ==============================================================================

    def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx

    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # ==============================================================================
    # Section 48: LOADING AND SAVING MODEL WEIGHTS IN PYTORCH
    # ==============================================================================

    model = GPTModel(GPT_CONFIG_124M)
    torch.save(model.state_dict(), "model.pth")

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        "model_and_optimizer.pth"
    )

    checkpoint = torch.load("model_and_optimizer.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train();

    # ==============================================================================
    # Section 49: LOADING PRETRAINED WEIGHTS FROM OPENAI
    # ==============================================================================

    pip install tensorflow>=2.15.0 tqdm>=4.66

    import tensorflow as tf
    import tqdm

    print("TensorFlow version:", tf.__version__)
    print("tqdm version:", tqdm.__version__)

    from gpt_download3 import download_and_load_gpt2

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())

    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)

    # Define model configurations in a dictionary for compactness
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval();

    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    import numpy as np

    def load_weights_into_gpt(gpt, params):
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


    load_weights_into_gpt(gpt, params)
    gpt.to(device);

    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # ==============================================================================
    # Section 50: FINETUNING FOR CLASSIFICATION
    # ==============================================================================

    # ==============================================================================
    # Section 51: DOWNLOADING DATASET
    # ==============================================================================

    import urllib.request
    import ssl
    import zipfile
    import os
    from pathlib import Path

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
        if data_file_path.exists():
            print(f"{data_file_path} already exists. Skipping download and extraction.")
            return

        # Create an unverified SSL context
        ssl_context = ssl._create_unverified_context()

        # Downloading the file
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

        # Unzipping the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        # Add .tsv file extension
        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        print(f"File downloaded and saved as {data_file_path}")

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    import pandas as pd

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    df

    print(df["Label"].value_counts())

    def create_balanced_dataset(df):

        # Count the instances of "spam"
        num_spam = df[df["Label"] == "spam"].shape[0]

        # Randomly sample "ham" instances to match the number of "spam" instances
        ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

        # Combine ham "subset" with "spam"
        balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

        return balanced_df

    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    def random_split(df, train_frac, validation_frac):
        # Shuffle the entire DataFrame
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)

        # Calculate split indices
        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)

        # Split the DataFrame
        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]

        return train_df, validation_df, test_df

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # Test size is implied to be 0.2 as the remainder

    print(len(train_df))
    print(len(validation_df))
    print(len(test_df))

    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    # ==============================================================================
    # Section 52: CREATING DATALOADERS
    # ==============================================================================

    import torch
    from torch.utils.data import Dataset


    class SpamDataset(Dataset):
        def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
            self.data = pd.read_csv(csv_file)

            # Pre-tokenize texts
            self.encoded_texts = [
                tokenizer.encode(text) for text in self.data["Text"]
            ]

            if max_length is None:
                self.max_length = self._longest_encoded_length()
            else:
                self.max_length = max_length

                # Truncate sequences if they are longer than max_length
                self.encoded_texts = [
                    encoded_text[:self.max_length]
                    for encoded_text in self.encoded_texts
                ]

            # Pad sequences to the longest sequence
            self.encoded_texts = [
                encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
                for encoded_text in self.encoded_texts
            ]

        def __getitem__(self, index):
            encoded = self.encoded_texts[index]
            label = self.data.iloc[index]["Label"]
            return (
                torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            )

        def __len__(self):
            return len(self.data)

        def _longest_encoded_length(self):
            max_length = 0
            for encoded_text in self.encoded_texts:
                encoded_length = len(encoded_text)
                if encoded_length > max_length:
                    max_length = encoded_length
            return max_length

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    print(train_dataset.max_length)

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    print(test_dataset.max_length)

    from torch.utils.data import DataLoader

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    # ==============================================================================
    # Section 53: INITIALIZING A MODEL WITH PRETRAINED WEIGHTS
    # ==============================================================================

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    from gpt_download3 import download_and_load_gpt2

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();

    text_1 = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))

    # ==============================================================================
    # Section 54: ADDING A CLASSIFICATION HEAD
    # ==============================================================================

    print(model)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)

    print("Last output token:", outputs[:, -1, :])

    # ==============================================================================
    # Section 55: CALCULATING THE CLASSIFICATION LOSS AND ACCURACY
    # ==============================================================================

    print("Last output token:", outputs[:, -1, :])

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    print(probas)
    label = torch.argmax(probas)
    print("Class label:", label.item())

    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())

    def calc_accuracy_loader(data_loader, model, device, num_batches=None):
        model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                with torch.no_grad():
                    logits = model(input_batch)[:, -1, :]  # Logits of last output token
                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
            else:
                break
        return correct_predictions / num_examples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note:
    # Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # As of this writing, in PyTorch 2.4, the results obtained via CPU and MPS were identical.
    # However, in earlier versions of PyTorch, you may observe different results when using MPS.

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    #print(f"Running on {device} device.")

    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

    torch.manual_seed(123) # For reproducibility due to the shuffling in the training data loader

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)[:, -1, :]  # Logits of last output token
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss

    # Same as in chapter 5
    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    # ==============================================================================
    # Section 56: FINETUNING THE MODEL ON SUPERVISED DATA
    # ==============================================================================

    # Overall the same as `train_model_simple` in chapter 5
    def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                                eval_freq, eval_iter):
        # Initialize lists to track losses and examples seen
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                examples_seen += input_batch.shape[0] # New: track examples instead of tokens 
                global_step += 1

                ## 130 batches: training, eval_Freq = 50 --> after 50 batches are processed in each epoch, we print train loss and val loss

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Calculate accuracy after each epoch
            train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
            val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
            print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy*100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

        return train_losses, val_losses, train_accs, val_accs, examples_seen

    # Same as chapter 5
    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

    import time

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    import matplotlib.pyplot as plt

    def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()

        # Create a second x-axis for examples seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Examples seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig(f"{label}-plot.pdf")
        plt.show()

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # ==============================================================================
    # Section 57: USING THE LLM AS A SPAM CLASSIFIER
    # ==============================================================================

    def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
        model.eval()

        # Prepare inputs to the model
        input_ids = tokenizer.encode(text)
        supported_context_length = model.pos_emb.weight.shape[0]
        # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
        # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

        # Truncate sequences if they too long
        input_ids = input_ids[:min(max_length, supported_context_length)]

        # Pad sequences to the longest sequence
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

        # Model inference
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return the classified result
        return "spam" if predicted_label == 1 else "not spam"

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    torch.save(model.state_dict(), "review_classifier.pth")

    model_state_dict = torch.load("review_classifier.pth")
    model.load_state_dict(model_state_dict)

    # ==============================================================================
    # Section 58: INSTRUCTION FINE-TUNING
    # ==============================================================================

    # ==============================================================================
    # Section 59: STEP 1: PREPARING DATASET
    # ==============================================================================

    import json
    import os
    import urllib
    import ssl

    def download_and_load_file(file_path, url):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        if not os.path.exists(file_path):
            with urllib.request.urlopen(url, context=ssl_context) as response:
                text_data = response.read().decode("utf-8")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data


    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    print("Example entry:\n", data[50])

    print("Another example entry:\n", data[999])

    # ==============================================================================
    # Section 60: CONVERTING INSTRUCTIONS INTO ALPACA FORMAT
    # ==============================================================================

    def format_input(entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"

    print(model_input + desired_response)

    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"

    print(model_input + desired_response)

    # ==============================================================================
    # Section 61: SPLITTING DATASET INTO TRAIN-TEST-VALIDATION
    # ==============================================================================

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    # ==============================================================================
    # Section 62: STEP 2: ORGANIZING DATA INTO TRAINING BATCHES
    # ==============================================================================

    import torch
    from torch.utils.data import Dataset


    class InstructionDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data

            # Pre-tokenize texts
            self.encoded_texts = []
            for entry in data:
                instruction_plus_input = format_input(entry)
                response_text = f"\n\n### Response:\n{entry['output']}"
                full_text = instruction_plus_input + response_text
                self.encoded_texts.append(
                    tokenizer.encode(full_text)
                )

        def __getitem__(self, index):
            return self.encoded_texts[index]

        def __len__(self):
            return len(self.data)

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    def custom_collate_draft_1(
        batch,
        pad_token_id=50256,
        device="cpu"
    ):
        # Find the longest sequence in the batch
        # and increase the max length by +1, which will add one extra
        # padding token below
        batch_max_length = max(len(item)+1 for item in batch)

        # Pad and prepare inputs
        inputs_lst = []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [pad_token_id]
            # Pad sequences to batch_max_length
            padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
            )
            # Via padded[:-1], we remove the extra padded token
            # that has been added via the +1 setting in batch_max_length
            # (the extra padding token will be relevant in later codes)
            inputs = torch.tensor(padded[:-1])
            inputs_lst.append(inputs)

        # Convert list of inputs to tensor and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        return inputs_tensor

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )

    print(custom_collate_draft_1(batch))

    # ==============================================================================
    # Section 63: CREATING TARGET TOKEN IDS FOR TRAINING
    # ==============================================================================

    def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        device="cpu"
    ):
        # Find the longest sequence in the batch
        batch_max_length = max(len(item)+1 for item in batch)

        # Pad and prepare inputs
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [pad_token_id]
            # Pad sequences to max_length
            padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
            )
            inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
            inputs_lst.append(inputs)
            targets_lst.append(targets)

        # Convert list of inputs to tensor and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        targets_tensor = torch.stack(targets_lst).to(device)
        return inputs_tensor, targets_tensor

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )

    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)

    def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
    ):
        # Find the longest sequence in the batch
        batch_max_length = max(len(item)+1 for item in batch)

        # Pad and prepare inputs and targets
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [pad_token_id]
            # Pad sequences to max_length
            padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
            )
            inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

            # New: Replace all but the first padding tokens in targets by ignore_index
            mask = targets == pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = ignore_index

            # New: Optionally truncate to maximum sequence length
            if allowed_max_length is not None:
                inputs = inputs[:allowed_max_length]
                targets = targets[:allowed_max_length]

            inputs_lst.append(inputs)
            targets_lst.append(targets)

        # Convert list of inputs and targets to tensors and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        targets_tensor = torch.stack(targets_lst).to(device)

        return inputs_tensor, targets_tensor

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )

    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    logits_1 = torch.tensor(
        [[-1.0, 1.0],  # 1st training example
         [-0.5, 1.5]]  # 2nd training example
    )
    targets_1 = torch.tensor([0, 1])


    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)

    logits_2 = torch.tensor(
        [[-1.0, 1.0],
         [-0.5, 1.5],
         [-0.5, 1.5]]  # New 3rd training example
    )
    targets_2 = torch.tensor([0, 1, 1])

    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)

    logits_2 = torch.tensor(
        [[-1.0, 1.0],
         [-0.5, 1.5],
         [-0.5, 1.5]]  # New 3rd training example
    )

    targets_3 = torch.tensor([0, 1, -100])

    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)

    # ==============================================================================
    # Section 64: MASKING TARGET TOKEN IDS
    # ==============================================================================

    # ==============================================================================
    # Section 65: STEP 3: CREATING DATALOADERS FOR AN INSTRUCTION DATASET
    # ==============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note:
    # Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is much faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # However, the resulting loss values may be slightly different.

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")

    print("Device:", device)

    from functools import partial
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    from torch.utils.data import DataLoader


    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    # ==============================================================================
    # Section 66: STEP 4: LOADING A PRETRAINED LLM
    # ==============================================================================

    from gpt_download3 import download_and_load_gpt2

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = generated_text[len(input_text):].strip()
    print(response_text)

    # ==============================================================================
    # Section 67: STEP 5: FINETUNING THE LLM ON INSTRUCTION DATA
    # ==============================================================================

    # ==============================================================================
    # Section 68: PREVIOUSLY DEFINED FUNCTIONS WHICH WE WILL REQUIRE
    # ==============================================================================

    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss


    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0: 
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

        return train_losses, val_losses, track_tokens_seen

    model.to(device)

    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    import time

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 1

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator


    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig("loss-plot.pdf")
        plt.show()

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # ==============================================================================
    # Section 69: STEP 6: EXTRACTING AND SAVING RESPONSES
    # ==============================================================================

    torch.manual_seed(123)


    for entry in test_data[:3]:

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
    )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

    from tqdm import tqdm

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text


    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

    print(test_data[0])

    import re


    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    # Load model via
    # model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

    # ==============================================================================
    # Section 70: STEP 7: EVALUATING THE FINE-TUNED LLM
    # ==============================================================================

    import psutil

    def check_if_running(process_name):
        running = False
        for proc in psutil.process_iter(["name"]):
            if process_name in proc.info["name"]:
                running = True
                break
        return running

    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    import urllib.request

    def query_model(
        prompt,
        model="llama3",
        url="http://localhost:11434/api/chat"
    ):
        # Create the data payload as a dictionary
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {     # Settings below are required for deterministic responses
                "seed": 123,
                "temperature": 0,
                "num_ctx": 2048
            }
        }


        # Convert the dictionary to a JSON formatted string and encode it to bytes
        payload = json.dumps(data).encode("utf-8")

        # Create a request object, setting the method to POST and adding necessary headers
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )
        request.add_header("Content-Type", "application/json")

        # Send the request and capture the response
        response_data = ""
        with urllib.request.urlopen(request) as response:
            # Read and decode the response
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json["message"]["content"]

        return response_data

    model = "llama3"
    result = query_model("What do Llamas eat?", model)
    print(result)

    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-------------------------")

    for entry in test_data[:2]:
        prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry['model_response']}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
        score = query_model(prompt, model)
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt, model))
        print("\n-------------------------")

    def generate_model_scores(json_data, json_key, model="llama3"):
        scores = []
        for entry in tqdm(json_data, desc="Scoring entries"):
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue

        return scores

    ## CONCLUSIONS


if __name__ == "__main__":
    main()
