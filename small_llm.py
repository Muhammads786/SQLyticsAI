import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SimpleLLM:
    """A simplified LLM that demonstrates the complete text generation process"""
    
    def __init__(self):
        # Vocabulary
        self.vocab = {"I": 1, "love": 2, "AI": 3, "data": 4, "apples": 5}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Embeddings (pretend these were learned during training)
        self.embedding_dim = 2
        self.embeddings = {
            1: np.array([1.0, 0.0]),    # "I"
            2: np.array([1.0, 1.0]),    # "love" 
            3: np.array([0.0, 1.0]),    # "AI"
            4: np.array([0.5, 0.5]),    # "data"
            5: np.array([0.2, 0.1])     # "apples"
        }
        
        # Positional encodings (extended for longer sequences)
        self.max_length = 10
        self.pos_embeddings = {
            i: np.array([0.1 * i, -0.1 * i]) for i in range(self.max_length)
        }
        
        # Output projection matrix (W_out)
        self.W_out = np.array([
            [2.0, 1.5, 1.8, 1.2, 0.5],  # Dimension 1 weights
            [0.5, 0.2, 1.0, 0.8, 0.1]   # Dimension 2 weights
        ])
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = text.split()
        return [self.vocab[token] for token in tokens if token in self.vocab]
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return " ".join([self.id_to_token[token_id] for token_id in token_ids])
    
    def context_encoding(self, token_ids: List[int]) -> np.ndarray:
        """Step 1: Convert tokens to embeddings + positional encoding"""
        print("=" * 60)
        print("STEP 1: CONTEXT ENCODING")
        print("=" * 60)
        
        encoded_tokens = []
        for i, token_id in enumerate(token_ids):
            token_emb = self.embeddings[token_id]
            pos_emb = self.pos_embeddings[i]
            combined = token_emb + pos_emb
            
            print(f"Token {i}: {self.id_to_token[token_id]} (ID: {token_id})")
            print(f"  Token embedding: {token_emb}")
            print(f"  Position {i} encoding: {pos_emb}")
            print(f"  Combined: {combined}")
            print()
            
            encoded_tokens.append(combined)
        
        X = np.array(encoded_tokens)
        print(f"Final input matrix X (shape {X.shape}):")
        print(X)
        print()
        return X
    
    def pattern_recognition(self, X: np.ndarray) -> np.ndarray:
        """Step 2: Self-attention mechanism"""
        print("=" * 60)
        print("STEP 2: PATTERN RECOGNITION (Self-Attention)")
        print("=" * 60)
        
        # Using identity matrices for Q, K, V projections
        Q = X  # Query
        K = X  # Key  
        V = X  # Value
        
        print("Q = K = V = X:")
        print(X)
        print()
        
        # Attention scores
        d_k = self.embedding_dim
        S = (Q @ K.T) / np.sqrt(d_k)
        
        print("Attention scores S = (Q @ Kᵀ) / √d_k:")
        print(S)
        print()
        
        # Apply causal mask (upper triangular = -inf)
        L = X.shape[0]
        mask = np.triu(np.ones((L, L)), k=1)
        S_masked = np.where(mask, -np.inf, S)
        
        print("After causal masking:")
        print(S_masked)
        print()
        
        # Softmax to get attention weights
        A = np.exp(S_masked - np.max(S_masked, axis=-1, keepdims=True))
        A = A / np.sum(A, axis=-1, keepdims=True)
        
        print("Attention weights A (softmax):")
        print(A)
        print()
        
        # Apply attention to values
        O = A @ V
        
        print("Output O = A @ V:")
        print(O)
        print()
        
        # For next token prediction, use last token's representation
        h = O[-1]  # Last token's contextualized representation
        print(f"Final contextual representation h (last token): {h}")
        print()
        
        return h
    
    def probability_distribution(self, h: np.ndarray) -> np.ndarray:
        """Step 3: Convert hidden state to probability distribution"""
        print("=" * 60)
        print("STEP 3: PROBABILITY DISTRIBUTION")
        print("=" * 60)
        
        # Compute logits
        logits = h @ self.W_out
        
        print("Hidden state h:", h)
        print("Output weights W_out:")
        print(self.W_out)
        print()
        
        print("Logits z = h @ W_out:")
        for i, (token, logit) in enumerate(zip(self.vocab.keys(), logits)):
            print(f"  {token:6s}: {logit:.4f}")
        print()
        
        # Softmax to get probabilities
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        print("Probabilities (softmax):")
        for i, (token, prob) in enumerate(zip(self.vocab.keys(), probs)):
            print(f"  {token:6s}: {prob:.4f} ({prob*100:.2f}%)")
        print()
        
        return probs
    
    def sampling(self, probs: np.ndarray, temperature: float = 1.0, 
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> int:
        """Step 4: Sample next token from probability distribution"""
        print("=" * 60)
        print("STEP 4: SAMPLING")
        print("=" * 60)
        
        original_probs = probs.copy()
        working_probs = probs.copy()
        
        # Apply temperature
        if temperature != 1.0:
            # Avoid division by zero and handle log(0)
            logits = np.log(np.clip(working_probs, 1e-10, 1.0))
            tempered_logits = logits / temperature
            working_probs = np.exp(tempered_logits - np.max(tempered_logits))
            working_probs = working_probs / np.sum(working_probs)
            
            print(f"After temperature {temperature}:")
            for i, (token, prob) in enumerate(zip(self.vocab.keys(), working_probs)):
                print(f"  {token:6s}: {prob:.4f}")
            print()
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(working_probs, -top_k)[-top_k:]
            mask = np.zeros_like(working_probs)
            mask[top_k_indices] = 1
            working_probs = working_probs * mask
            working_probs = working_probs / np.sum(working_probs)
            
            print(f"After top-{top_k} filtering:")
            kept_tokens = [self.id_to_token[i+1] for i in range(len(working_probs)) if working_probs[i] > 0]
            print(f"  Kept tokens: {kept_tokens}")
            for i, (token, prob) in enumerate(zip(self.vocab.keys(), working_probs)):
                if prob > 0:
                    print(f"  {token:6s}: {prob:.4f}")
            print()
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(working_probs)[::-1]
            sorted_probs = working_probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find cutoff where cumulative probability exceeds top_p
            cutoff_mask = cumulative_probs <= top_p
            if not np.any(cutoff_mask):
                cutoff_index = 1  # Keep at least one token
            else:
                cutoff_index = np.where(cutoff_mask)[0][-1] + 1
            
            # Keep only top-p tokens
            mask = np.zeros_like(working_probs)
            mask[sorted_indices[:cutoff_index]] = 1
            working_probs = working_probs * mask
            working_probs = working_probs / np.sum(working_probs)
            
            print(f"After top-p (p={top_p}) filtering:")
            kept_tokens = [self.id_to_token[i+1] for i in range(len(working_probs)) if working_probs[i] > 0]
            print(f"  Kept tokens: {kept_tokens}")
            for i, (token, prob) in enumerate(zip(self.vocab.keys(), working_probs)):
                if prob > 0:
                    print(f"  {token:6s}: {prob:.4f}")
            print()
        
        # Sample from final distribution
        next_token_id = np.random.choice(len(working_probs), p=working_probs) + 1
        
        print("Sampling result:")
        print(f"  Original highest: {self.id_to_token[np.argmax(original_probs) + 1]}")
        print(f"  Final distribution: {[f'{p:.3f}' for p in working_probs]}")
        print(f"  Sampled token: {self.id_to_token[next_token_id]}")
        print()
        
        return next_token_id
    
    def generate(self, prompt: str, max_new_tokens: int = 3, **sampling_kwargs) -> str:
        """Complete text generation pipeline - FIXED VERSION"""
        print("🚀 GENERATING TEXT WITH SIMPLE LLM")
        print("Prompt:", prompt)
        print()
        
        # Tokenize prompt
        token_ids = self.tokenize(prompt)
        generated_ids = token_ids.copy()
        
        for step in range(max_new_tokens):
            print(f"🎯 GENERATION STEP {step + 1}")
            print("-" * 40)
            print(f"Current sequence: {self.detokenize(generated_ids)}")
            print(f"Token IDs: {generated_ids}")
            print()
            
            # Step 1: Context Encoding - Use ENTIRE sequence so far
            X = self.context_encoding(generated_ids)
            
            # Step 2: Pattern Recognition  
            h = self.pattern_recognition(X)
            
            # Step 3: Probability Distribution
            probs = self.probability_distribution(h)
            
            # Step 4: Sampling
            next_token_id = self.sampling(probs, **sampling_kwargs)
            
            # Append to the entire sequence (not replace!)
            generated_ids.append(next_token_id)
            
            current_text = self.detokenize(generated_ids)
            print(f"✅ Generated: '{current_text}'")
            print("\n" + "="*80 + "\n")
            
            # Early stopping if we exceed max length
            if len(generated_ids) >= self.max_length:
                break
        
        return self.detokenize(generated_ids)

# Demo the complete process
def demo_llm_generation():
    llm = SimpleLLM()
    
    print("🔍 VOCABULARY AND EMBEDDINGS")
    print("Vocabulary:", llm.vocab)
    print("Embeddings:")
    for token, emb in llm.embeddings.items():
        print(f"  {llm.id_to_token[token]:6s}: {emb}")
    print()
    
    # Test different sampling strategies
    prompt = "I love"
    
    print("1. GREEDY SAMPLING (temperature=0.1)")
    result1 = llm.generate(prompt, max_new_tokens=2, temperature=0.1)
    print(f"Final: {result1}\n")
    
    print("2. CREATIVE SAMPLING (temperature=1.5, top_k=3)")
    result2 = llm.generate(prompt, max_new_tokens=2, temperature=1.5, top_k=3)
    print(f"Final: {result2}\n")
    
    print("3. NUCLEUS SAMPLING (top_p=0.8)")
    result3 = llm.generate(prompt, max_new_tokens=2, top_p=0.8)
    print(f"Final: {result3}\n")

if __name__ == "__main__":
    demo_llm_generation()