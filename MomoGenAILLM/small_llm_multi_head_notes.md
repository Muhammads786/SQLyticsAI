In last two articles, we unpacked the mathematical and technical core of large language models which is a simple and beautiful idea of predicting the next token. Today, we’re building a truely trained mini-llm from scratch,  so you can feel how probabilities turn into words, and how tiny design choices shape the model’s voice. It will also show you the internal mechanics of LLM Models and how they are solving advanced problems.

Most people see Large Language Models (LLMs) as magical black boxes as they type a prompt, get a fluent answer, and wonder how the machine actually thinks.

But behind the scenes, every LLM (from ChatGPT to Gemini)  is built on a handful of beautifully simple mathematical principles: 

Eembeddings, 

Matrix Multiplications, 

Probability, 

and Gradient descent.

So instead of just talking about it, Lets build to get a deep understanding of inner world of LLM.

Introducing the TrulyTrainedLLM
The TrulyTrainedLLM is a miniature but fully functional LLM built in pure Python. It doesn’t only mimic pattern-matching but also learns through gradient descent, just like a neural network. Following is the workflow to show the internal mechanics of our LLM before moving towards technical Mini-LLM from Scratchdetails.


Workflow of  Mini-LLM from Scratch


Lets Explore the above workflow step by step and you will understand LLM internals at the end. 

demonstrate_real_training
Main demonstration function that showcases the fully trained neural network's capabilities by testing it on carefully selected prompts from the training distribution.

def demonstrate_real_training():
    print("🎯 NEURAL NETWORK TRAINING DEMONSTRATION")
    print("=" * 60)
    print("This model uses ACTUAL gradient descent training")
    print("like real neural networks, not just pattern matching.")
    print("=" * 60)
    print()
    
    llm = TrulyTrainedLLM(vocab_size=150, embedding_dim=16, learning_rate=0.02)
    
    print("\n🧪 TESTING TRAINED PATTERNS:")
    print("=" * 60)
    
    test_prompts = [
        "I love",
        "The AI", 
        "Machine learning",
        "We need to",
        "The research",
        "Artificial intelligence",
        "Scientists",
        "Education"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. ", end="")
        llm.generate(prompt, max_new_tokens=6, temperature=0.5)
        print()

Phase 1: Setup & Initialization
```python
# Initialize the trained LLM with optimized parameters
llm = TrulyTrainedLLM(vocab_size=150, embedding_dim=16, learning_rate=0.02)
Creates a pre-trained model (training happens during initialization)

Uses optimized hyperparameters for better performance

Phase 2: Test Prompt Definition
```python
test_prompts = [
    "I love", "The AI", "Machine learning", "We need to",
    "The research", "Artificial intelligence", "Scientists", "Education"
]
```
Carefully curated prompts that match training corpus patterns

Each prompt tests the model's ability to continue learned patterns

Covers all major semantic categories from training data

Phase 3: Generation Loop
```python
for i, prompt in enumerate(test_prompts, 1):
    llm.generate(prompt, max_new_tokens=6, temperature=0.5)
```
Iterates through all test prompts with numbered output

Uses conservative temperature (0.5) for focused, coherent generation

Limits to 6 new tokens to demonstrate pattern completion without rambling

Demonstration Value
Shows Real Learning

Proves the model learned actual linguistic patterns, not just memorization

Demonstrates context-aware continuation based on training

Shows semantic understanding within the trained domain

Educational Purpose

Visual proof of gradient descent effectiveness

Comparable outputs across similar prompt categories

Controlled environment to observe neural network behavior

Expected Output

The function produces a clean, educational demonstration showing how the model coherently continues each prompt based on patterns learned during actual neural network training.

Output Format:
```
🧪 TESTING TRAINED PATTERNS:
============================================================
1. 🧠 PROMPT: 'I love'
   ==================================================
   🤔 GENERATING: [continuation words...]
   💡 RESULT: [complete generated sentence]

```




Code Output
Pre-Requisites for Running Generate() function
generate() function is the key driver of predicting next token. However, following are the key pre-requisistes required to complete before calling it. We are completing them as part of init() function of TruelyTrainLLM class in our code 

Build Vocabulary

Initialize Embedding 

Initialize Weights

Train with Gradient Descent

Create Positional Encodings

Build Vocabulary
Constructs the vocabulary mapping from the training corpus by tokenizing all text, counting word frequencies, and creating bidirectional token-ID mappings with special tokens for padding and unknown words.







Add a caption (optional)


```python
def buildvocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]

```
Returns: Tuple of (vocab, id_to_token) where:
- vocab: Token → ID mapping dictionary
- id_to_token: ID → Token mapping dictionary


Key Processing Steps

Step 1: Tokenization

```python
tokens = re.findall(r'\b\w+\b', text.lower())
```
Regex Pattern: \b\w+\b matches complete words only

Lowercase Conversion: Normalizes all text for consistent vocabulary

No Punctuation: Simple word-only tokenization

Step 2: Frequency Analysis

```python
token_counts = Counter(all_tokens)
most_common = token_counts.most_common(self.vocab_size - 2)
```
Counts all token occurrences across corpus

Selects top N tokens based on vocab_size (reserving 2 for special tokens)

Step 3: Vocabulary Construction

```python

vocab = {"<PAD>": 0, "<UNK>": 1}
id_to_token = {0: "<PAD>", 1: "<UNK>"}
```
Special Tokens:

<PAD>: Padding token (ID: 0)

<UNK>: Unknown token (ID: 1)

Step 4: Token Mapping

```python
for i, (token, count) in enumerate(most_common):
    vocab[token] = i + 2
    id_to_token[i + 2] = token
```

Assigns sequential IDs starting from 2

Creates bidirectional mappings for easy lookup

Output Example

```
Top 10 tokens: ['the', 'ai', 'i', 'learning', 'love', 'machine', 'research', 'intelligence', 'artificial', 'scientists']
```
Resulting Vocabulary Structure:

```python

vocab = {
    '<PAD>': 0, '<UNK>': 1,
    'the': 2, 'ai': 3, 'i': 4, 'learning': 5, 
    'love': 6, 'machine': 7, ...

}

id_to_token = {
    0: '<PAD>', 1: '<UNK>', 
    2: 'the', 3: 'ai', 4: 'i', 5: 'learning',
    6: 'love', 7: 'machine', ...

}

```


Initialize Embedding 
Creates the initial embedding matrix that will map each vocabulary token to a dense vector representation in the neural network's embedding space.

```python

def _initialize_embeddings(self) -> np.ndarray:
        """Initialize embedding matrix"""
        return np.random.normal(0, 0.1, (len(self.vocab), self.embedding_dim))
Mean: 0 - Centers embeddings around zero

Standard Deviation: 0.1 - Small values to prevent large initial gradients

Shape: (vocab_size, embedding_dim) - Matrix mapping each token to a vector

Example Output Shape:

If vocab_size = 150 and embedding_dim = 16

Returns matrix of shape (150, 16)

Each of the 150 tokens gets a 16-dimensional vector

Why This Matters

Embedding Matrix Role:

Token Representation: Converts discrete token IDs to continuous vectors

Learnable Features: These vectors get updated during training to capture semantic meaning

Neural Network Input: Serves as the first layer of the model

Initialization Strategy:

Small Random Values: Prevents saturation in activation functions

Normal Distribution: Common practice in deep learning

Controlled Scale: 0.1 std dev keeps gradients stable during early training

In Context:

This embedding matrix becomes the foundation that gradient descent will optimize during training, allowing the model to learn meaningful semantic relationships between words based on the training corpus patterns.

Simple but crucial - this random initialization is where the learning journey begins!

Initialize Weights

def _initialize_weights(self, shape):
        """Xavier initialization"""
        bound = math.sqrt(6.0 / sum(shape))
        return np.random.uniform(-bound, bound, shape)
Implements Xavier/Glorot initialization for neural network weights, which helps maintain stable gradients during training by keeping activation variances consistent across layers.

Xavier Initialization Formula

```python
bound = math.sqrt(6.0 / sum(shape))
return np.random.uniform(-bound, bound, shape)
Calculation: 

For shape (input_dim, output_dim)

bound = √(6 / (input_dim + output_dim))

Weights sampled uniformly from [-bound, +bound]

Weight Initialization Methods Comparison





Add a caption (optional)
Why Xavier for This LLM

Perfect Fit for This Architecture:

Embedding Dimensions: 16 (relatively small)

Activation Patterns: Implicit tanh-like behavior in embeddings

Stable Gradients: Prevents vanishing/exploding gradients in deep networks

Mathematical Foundation:

Variance Preservation: Ensures consistent signal flow forward and backward

Theoretical Backing: Derived to maintain activation variances across layers

Proven Performance: Industry standard for tanh/sigmoid networks

In Context:

This initialization is used for the output layer weights (`W_out`) that map from embedding space back to vocabulary space, ensuring stable training from the very first gradient step!

Xavier initialization gives our LLM a solid mathematical foundation for effective learning!

Train with Gradient Descent
The core training engine that implements actual neural network training using vanilla gradient descent. This function performs the complete training loop: forward pass, loss calculation, backward pass, and weight updates.





Add a caption (optional)
def _train_with_gradient_descent(self):
        """ACTUAL neural network training with gradient descent"""
        print("🧠 TRAINING WITH GRADIENT DESCENT...")
        
        # Convert training data to (input, target) pairs
        training_pairs = []
        for text in self.training_corpus:
            tokens = re.findall(r'\b\w+\b', text.lower())
            token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
            
            for i in range(len(token_ids) - 1):
                input_ids = token_ids[:i+1]
                target_id = token_ids[i+1]
                training_pairs.append((input_ids, target_id))
        
        # Training loop
        for epoch in range(5000):  # Multiple epochs
            total_loss = 0
            np.random.shuffle(training_pairs)
            
            for input_ids, target_id in training_pairs[:150]:  # Use subset for speed
                # Forward pass
                context_embedding = np.mean([self.embeddings[token_id] for token_id in input_ids[-3:]], axis=0)
                logits = context_embedding @ self.W_out
                probs = self._softmax(logits)
                
                # Calculate loss (cross entropy)
                loss = -np.log(probs[target_id] + 1e-8)
                total_loss += loss
                
                # Backward pass - ACTUAL GRADIENT DESCENT
                # Gradient of loss w.r.t. logits
                d_logits = probs.copy()
                d_logits[target_id] -= 1
                
                # Gradients for W_out
                d_W_out = np.outer(context_embedding, d_logits)
                
                # Gradient for embeddings (simplified)
                d_context = self.W_out @ d_logits
                
                # Update weights - ACTUAL GRADIENT DESCENT
                self.W_out -= self.learning_rate * d_W_out
                
                # Update embeddings of context tokens
                for token_id in input_ids[-3:]:
                    self.embeddings[token_id] -= self.learning_rate * 0.1 * d_context
            
            if epoch % 25 == 0:
                print(f"  Epoch {epoch + 1}, Loss: {total_loss/len(training_pairs):.4f}")
        
        print("✅ Training completed!")
Training Pipeline Breakdown

Phase 1: Data Preparation

```python
# Create (input, target) pairs for next-word prediction

for i in range(len(token_ids) - 1):
    input_ids = token_ids[:i+1]    # Context
    target_id = token_ids[i+1]     # Next word to predict

```


Phase 2: Training Loop

# Training loop
        for epoch in range(5000):  # Multiple epochs
            total_loss = 0
            np.random.shuffle(training_pairs)
5000 epochs of full training cycles

Shuffling for better convergence

Batch subset (150 pairs) for computational efficiency

Phase 3: Forward Pass

```python
context_embedding = np.mean([self.embeddings[token_id] for token_id in input_ids[-3:]], axis=0)
logits = context_embedding @ self.W_out
probs = self._softmax(logits)
Context window: Last 3 tokens

Embedding average: Simple but effective context representation

Softmax: Converts logits to probabilities

Phase 4: Backward Pass & Updates

```python
# Gradient computation
d_logits = probs.copy()
d_logits[target_id] -= 1
d_W_out = np.outer(context_embedding, d_logits)
# Weight updates
self.W_out -= self.learning_rate  d_W_out
self.embeddings[token_id] -= self.learning_rate  0.1  d_context

Gradient Descent Variants Comparison





Add a caption (optional)
Key Design Choices

Context Window (3 tokens):

Computational Efficiency: Limits sequence length

Pattern Focus: Training corpus has clear short-range dependencies

Memory Constraints: Educational implementation

Learning Rate Strategy:

Fixed Rate: Simple and predictable

Embedding Discount (0.1): Slower embedding updates for stability

Loss Calculation:

Cross-Entropy: Standard for classification

Numerical Stability: + 1e-8 prevents log(0)

Performance Characteristics

Training Speed: Fast due to small scale

Convergence: Reliable for clear patterns

Memory Usage: Minimal - no optimizer state

This implementation proves that even basic gradient descent can learn meaningful language patterns when the training data has clear structure!

Create Positional Encodings
Implements sinusoidal positional encodings to give the model information about token positions in sequences, following the original Transformer architecture pattern.

Sinusoidal Encoding Formula

Core Mathematical Pattern:

```python
encoding = np.array([
    math.sin(pos / 10000  (i / self.embedding_dim)) if i % 2 == 0 
    else math.cos(pos / 10000  ((i-1) / self.embedding_dim))
    for i in range(self.embedding_dim)
])

How It Works:

Even indices (i % 2 == 0): sin(pos / 10000^(i/d))

Odd indices (i % 2 == 1): cos(pos / 10000^((i-1)/d))

Frequency decreases across dimensions

Unique pattern for each position

Key Design Choices

1. Scale Factor (0.1):

```python
pos_embeddings[pos] = encoding  0.1
```
Small positional effect: Doesn't dominate word embeddings

Balanced influence: Position matters but content matters more

Training stability: Prevents large positional biases

2. Frequency Scaling:

Base 10000: Controls how quickly frequencies change across dimensions

Dimension-aware: i/embedding_dim normalizes across vector size

Geometric progression: Creates diverse frequency components

Why Sinusoidal Encodings?

Advantages:

Absolute & Relative: Captures both exact positions and relative distances

Deterministic: No learned parameters, consistent across runs

Extrapolation: Can handle sequences longer than training max_length

Unique encoding: Every position has distinct signature

Example Output:

For position 0, embedding_dim=4:

```
[sin(0/10000^0), cos(0/10000^0), sin(0/10000^(2/4)), cos(0/10000^(2/4))]  0.1
= [0.0, 0.1, 0.0, 0.1]  # Then scaled by 0.1

```


In Context Usage:

```python
# During generation, add positional encoding to context
context_embedding += self.pos_embeddings[pos]
```


This gives the model position awareness while maintaining the simplicity of our educational implementation - a scaled-down version of what powers major LLMs like GPT and BERT!

Predict Next Token
The core inference engine that predicts the next token using the trained neural network with advanced sampling techniques for high-quality text generation.





Add a caption (optional)




Add a caption (optional)
Multi-Stage Prediction Pipeline

Stage 1: Context Processing

```python
# Context window: last 3 tokens
context_window = context_ids[-3:] if len(context_ids) > 3 else context_ids

# Average embeddings for context
context_embedding = np.mean([self.embeddings[token_id] for token_id in context_window], axis=0)

# Add positional encoding
context_embedding += self.pos_embeddings[len(context_ids)]

```
Fixed context window: 3 tokens for consistency

Embedding averaging: Simple but effective context representation

Position awareness: Adds positional encodings

Stage 2: Neural Network Inference

```python
logits = context_embedding @ self.W_out
```


Matrix multiplication: Embedding → vocabulary space

Raw scores: Unnormalized logits for each vocabulary token

Stage 3: Advanced Sampling Techniques

Temperature Scaling (0.7)

```python
logits = logits / temperature
```
Conservative temperature: 0.7 for focused, coherent output

Lower = more deterministic, Higher = more creative

Repetition Penalty

```python
for token_id in recent_tokens:
    logits[token_id] -= 3.0
```
Penalizes recent tokens: Prevents repetitive loops

Strong penalty: -3.0 significantly reduces probability

Context window: Last 4 tokens considered "recent"

Top-p (Nucleus) Filtering (0.9)

```python
cutoff_index = np.where(cumulative_probs >= 0.9)[0]
```
Keeps top 90% of probability mass

Dynamic vocabulary: Size varies based on distribution

Eliminates low-probability tails

Sampling Strategy Comparison





Add a caption (optional)
Why This Sophisticated Approach?

Educational Value

Shows real LLM techniques used in production models

Demonstrates sampling tradeoffs between creativity and coherence

Teaches advanced inference beyond simple argmax

Technical Benefits:

Controlled randomness: Temperature balances predictability/creativity

Anti-repetition: Prevents common generation artifacts

Quality filtering: Top-p ensures only reasonable tokens are considered

Output Quality Features:

Coherent continuations: Maintains context and style

Diverse but sensible: Avoids completely random outputs

Pattern-consistent: Follows training corpus patterns

Repetition-free: Natural flow without getting stuck

This function transforms raw neural network outputs into human-like text generation using the same advanced techniques that power major AI systems!


