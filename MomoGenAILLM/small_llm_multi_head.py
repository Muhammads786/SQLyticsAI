import numpy as np
import re
from collections import Counter
import math
from typing import List, Dict, Tuple

class TrulyTrainedLLM:
    """LLM with ACTUAL neural network training using gradient descent"""
    
    def __init__(self, vocab_size=200, embedding_dim=16, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.max_length = 128
        
        # High-quality training corpus with VERY clear patterns
        self.training_corpus = self._get_super_clear_corpus()
        
        # Build vocabulary
        self.vocab, self.id_to_token = self._build_vocabulary()
        
        # Initialize weights PROPERLY
        self.embeddings = self._initialize_embeddings()
        self.W_out = self._initialize_weights((embedding_dim, len(self.vocab)))
        
        # ACTUAL TRAINING with gradient descent
        self._train_with_gradient_descent()
        
        self.pos_embeddings = self._create_positional_encodings()
        
        print("🚀 TRULY TRAINED LLM READY!")
        print(f"• Vocabulary: {len(self.vocab)} tokens")
        print(f"• Training iterations: 1000+ gradient steps")
        print()
    
    def _get_super_clear_corpus(self) -> List[str]:
        """Extremely clear training corpus with obvious patterns"""
        return [
            # CRYSTAL CLEAR PATTERNS - No ambiguity
            "I love learning about artificial intelligence",
            "I love studying machine learning algorithms", 
            "I love building intelligent computer systems",
            "I love creating new AI applications",
            "I love developing smart software programs",
            
            "The AI system processes information quickly",
            "The AI model learns from data patterns",
            "The AI algorithm solves complex problems",
            "The AI technology helps many people",
            "The AI research advances science greatly",
            
            "Machine learning recognizes patterns well",
            "Machine learning predicts future outcomes", 
            "Machine learning improves with practice",
            "Machine learning solves hard problems",
            "Machine learning helps businesses grow",
            
            "We need to understand the data first",
            "We need to build better models now",
            "We need to test our systems thoroughly",
            "We need to learn new methods always",
            "We need to improve our skills constantly",
            
            "The research shows positive results clearly",
            "The research demonstrates new possibilities",
            "The research proves the method works",
            "The research confirms our hypothesis",
            "The research advances our knowledge",
            
            "Artificial intelligence transforms industries",
            "Artificial intelligence creates new jobs",
            "Artificial intelligence solves problems",
            "Artificial intelligence helps humanity",
            "Artificial intelligence advances science",
            
            "Scientists discover new things daily",
            "Scientists work hard on research",
            "Scientists publish important papers",
            "Scientists solve difficult problems",
            "Scientists advance human knowledge",
            
            "Education improves people's lives",
            "Education creates opportunities for all",
            "Education develops important skills",
            "Education builds better societies",
            "Education empowers future generations",
        ]
    
    def _build_vocabulary(self) -> tuple:
        """Build vocabulary from corpus"""
        all_tokens = []
        for text in self.training_corpus:
            tokens = re.findall(r'\b\w+\b', text.lower())  # Only words, no punctuation for simplicity
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(self.vocab_size - 2)  # Only UNK and PAD
        
        vocab = {"<PAD>": 0, "<UNK>": 1}
        id_to_token = {0: "<PAD>", 1: "<UNK>"}
        
        for i, (token, count) in enumerate(most_common):
            vocab[token] = i + 2
            id_to_token[i + 2] = token
        
        print(f"Top 10 tokens: {[token for token, count in most_common[:10]]}")
        return vocab, id_to_token
    
    def _initialize_embeddings(self) -> np.ndarray:
        """Initialize embedding matrix"""
        return np.random.normal(0, 0.1, (len(self.vocab), self.embedding_dim))
    
    def _initialize_weights(self, shape):
        """Xavier initialization"""
        bound = math.sqrt(6.0 / sum(shape))
        return np.random.uniform(-bound, bound, shape)
    
    def _softmax(self, x):
        """Stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
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
    
    def _create_positional_encodings(self) -> Dict[int, np.ndarray]:
        """Positional encodings"""
        pos_embeddings = {}
        for pos in range(self.max_length):
            encoding = np.array([math.sin(pos / 10000 ** (i / self.embedding_dim)) 
                               if i % 2 == 0 else math.cos(pos / 10000 ** ((i-1) / self.embedding_dim))
                               for i in range(self.embedding_dim)])
            pos_embeddings[pos] = encoding * 0.1  # Small positional effect
        return pos_embeddings
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to readable text"""
        tokens = [self.id_to_token.get(token_id, "<UNK>") for token_id in token_ids]
        # Capitalize first word
        if tokens:
            tokens[0] = tokens[0].capitalize()
        return ' '.join(tokens)
    
    def _predict_next_token(self, context_ids: List[int], temperature: float = 0.7) -> int:
        """Predict next token given context"""
        # Use last 3 tokens as context
        if len(context_ids) > 3:
            context_window = context_ids[-3:]
        else:
            context_window = context_ids
        
        # Context embedding (average)
        context_embedding = np.mean([self.embeddings[token_id] for token_id in context_window], axis=0)
        
        # Add positional information
        pos = len(context_ids)
        if pos < len(self.pos_embeddings):
            context_embedding += self.pos_embeddings[pos]
        
        # Get logits
        logits = context_embedding @ self.W_out
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply repetition penalty
        recent_tokens = set(context_ids[-4:])
        for token_id in recent_tokens:
            logits[token_id] -= 3.0
        
        # Convert to probabilities
        probs = self._softmax(logits)
        
        # Apply top-p filtering
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        cutoff_index = np.where(cumulative_probs >= 0.9)[0]
        if len(cutoff_index) > 0:
            cutoff_index = cutoff_index[0] + 1
            final_probs = np.zeros_like(probs)
            final_probs[sorted_indices[:cutoff_index]] = probs[sorted_indices[:cutoff_index]]
            final_probs = final_probs / np.sum(final_probs)
        else:
            final_probs = probs
        
        # Sample next token
        return np.random.choice(len(final_probs), p=final_probs)
    
    def generate(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.6) -> str:
        """Generate coherent text"""
        print(f"🧠 PROMPT: '{prompt}'")
        print("=" * 50)
        
        token_ids = self.tokenize(prompt)
        generated_ids = token_ids.copy()
        
        print("🤔 GENERATING: ", end="")
        
        for step in range(max_new_tokens):
            next_token_id = self._predict_next_token(generated_ids, temperature)
            generated_ids.append(next_token_id)
            
            current_word = self.id_to_token.get(next_token_id, "<UNK>")
            print(f"{current_word} ", end="")
            
            # Stop if we start repeating or reach max length
            if len(generated_ids) >= 15:
                break
            if len(set(generated_ids[-4:])) < 2:  # Stop if repeating
                break
        
        final_result = self.detokenize(generated_ids)
        print(f"\n💡 RESULT: {final_result}")
        print("=" * 50)
        return final_result

def demonstrate_real_training():
    print("🎯 NEURAL NETWORK TRAINING DEMONSTRATION")
    print("=" * 60)
    print("This model uses ACTUAL gradient descent training")
    print("like real neural networks, not just pattern matching.")
    print("=" * 60)
    print()
    
    llm = TrulyTrainedLLM(vocab_size=150, embedding_dim=16, learning_rate=0.01)
    
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

if __name__ == "__main__":
    demonstrate_real_training()