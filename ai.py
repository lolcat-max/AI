import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from collections import Counter, defaultdict
import hashlib
import re  # Added for rule parsing and judgment

# ================================================================
# NEW: EMBEDDED TEMPLATE ENGINE
# ================================================================

class OrigamiLLM:
    def __init__(self):
        # Training corpus about origami and neural networks
        filename = input("Filename: ")
        with open(filename, encoding="utf-8") as f:
            text = f.read()
        self.corpus = text.lower().split(".")
        
        # Build n-gram model (bigrams and trigrams)
        self.bigrams = {}
        self.trigrams = {}
        self.build_model()

        # NEW: Initialize and build embedded templates
        self.template_matcher = EmbeddedTemplateMatcher()
        self.template_matcher.build_templates_from_corpus(self.corpus)

    def build_model(self):
        """Build n-gram probability model"""
        for sentence in self.corpus:
            words = sentence.split()
            if len(words) < 3: continue
            
            # Build bigrams
            for i in range(len(words) - 1):
                if words[i] not in self.bigrams:
                    self.bigrams[words[i]] = []
                self.bigrams[words[i]].append(words[i + 1])
            
            # Build trigrams
            for i in range(len(words) - 2):
                key = (words[i], words[i + 1])
                if key not in self.trigrams:
                    self.trigrams[key] = []
                self.trigrams[key].append(words[i + 2])

    def generate_text(self, start_word=None, max_words=15, stack_state=0.0, chirality_strength=0.0):
        """
        Generate text based on stack state and template tuning.
        Modified: stack_state controls model usage, chirality_strength introduces handed bias in selection.
        """
        
        if start_word is None or start_word not in self.bigrams:
            start_word = random.choice(list(self.bigrams.keys()))
        
        result = [start_word]
        current_word = start_word
        
        for _ in range(max_words - 1):
            use_model = random.random() < stack_state
            candidates = []
            
            if use_model:
                # Use learned n-gram model
                trigram_key = tuple(result[-2:])
                if len(result) >= 2 and trigram_key in self.trigrams:
                    candidates = self.trigrams[trigram_key]
                elif current_word in self.bigrams:
                    candidates = self.bigrams[current_word]

            if not candidates:
                # Fallback to random selection
                if self.bigrams:
                    candidates = [random.choice(list(self.bigrams.keys())) for _ in range(5)]
           
            # --- TEMPLATE TUNING (as before) ---
            context = ' '.join(result[-5:]) # Use last 5 words as context
            best_template_tuple = self.template_matcher.find_best_template(context)
            
            probs = np.ones(len(candidates)) # Start with uniform probability
            
            if chirality_strength > 0 and best_template_tuple is not None:
                template_text, _ = best_template_tuple
                template_words = template_text.split()
                
                # Find where the context might fit in the template
                try:
                    # Find last word of context in template
                    idx = template_words.index(result[-1])
                    if idx < len(template_words) - 1:
                        template_next_word = template_words[idx + 1]
                        
                        # If the template's suggestion is a candidate, boost its probability
                        if template_next_word in candidates:
                            boost_idx = candidates.index(template_next_word)
                            # Boost is proportional to chirality strength, with handedness twist
                            twist_factor = 1 + np.sin(np.pi * chirality_strength)  # Sinusoidal twist for handedness
                            probs[boost_idx] += 10.0 * chirality_strength * twist_factor
                except (ValueError, IndexError):
                    pass # Context not in template or at the end

            # Normalize probabilities
            probs /= probs.sum() if probs.sum() > 0 else 1
            
            # Choose next word based on potentially tuned probabilities
            if candidates:
                next_word = np.random.choice(candidates, p=probs)
            else:
                next_word = random.choice(list(self.bigrams.keys())) if self.bigrams else "end"
            
            result.append(next_word)
            current_word = next_word
            
        generated = ' '.join(result)
        
        return generated

class EmbeddedTemplateMatcher:
    """
    Finds and uses semantic templates derived from the corpus
    to guide text generation.
    """
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim
        self.templates = []  # List of (template_text, template_vector) tuples
        self.word_vectors = {}
   
    def _get_word_vector(self, word: str) -> np.ndarray:
        """Generate a deterministic vector for a word using hashing."""
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # Use SHA-256 to create a deterministic, high-dimensional vector
        hash_bytes = hashlib.sha256(word.encode('utf-8')).digest()
        seed = int.from_bytes(hash_bytes[:4], 'little')
        rng = np.random.default_rng(seed)
        vec = rng.random(self.vector_dim) - 0.5
        vec /= np.linalg.norm(vec) # Normalize
        self.word_vectors[word] = vec
        return vec

    def _get_sentence_vector(self, sentence: str) -> np.ndarray:
        """Convert a sentence to a vector by averaging word vectors."""
        words = sentence.split()
        if not words:
            return np.zeros(self.vector_dim)
        
        vectors = [self._get_word_vector(w) for w in words]
        avg_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(avg_vec)
        return avg_vec / norm if norm > 0 else np.zeros(self.vector_dim)

    def build_templates_from_corpus(self, corpus: list, num_templates: int = 50):
        """Derive templates from the corpus."""
        # Select representative sentences as templates
        step = max(1, len(corpus) // num_templates)
        for i in range(0, len(corpus), step):
            sentence = corpus[i]
            if len(sentence.split()) > 5: # Ensure template has some substance
                template_vector = self._get_sentence_vector(sentence)
                self.templates.append((sentence, template_vector))

    def find_best_template(self, context: str) -> tuple | None:
        """Find the most semantically similar template for the given context."""
        if not self.templates or not context:
            return None
        
        context_vector = self._get_sentence_vector(context)
        
        best_similarity = -1
        best_template = None
        
        # Calculate cosine similarity to all templates
        template_vectors = np.array([t[1] for t in self.templates])
        similarities = np.dot(template_vectors, context_vector)
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > 0.4: # Similarity threshold
            return self.templates[best_idx]
        return None

# Chiral stacked 2D geometry functions (replacing origami)
def create_2d_layer(num_points=100, size=1.0, layer_id=0):
    """
    Create a 2D triangular lattice layer for stacking.
    """
    # Generate points in a triangular lattice within a square
    x = np.linspace(-size/2, size/2, int(np.sqrt(num_points)))
    y = np.linspace(-size/2, size/2, int(np.sqrt(num_points)))
    X, Y = np.meshgrid(x, y)
    # Offset every other row for triangular lattice
    Y[::2] += (y[1] - y[0]) / 2
    points = np.column_stack((X.ravel()[:num_points], Y.ravel()[:num_points]))
    # Rotate by layer_id * base_twist for initial chirality
    theta = np.deg2rad(layer_id * 30)  # 30 degrees per layer for twist
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = points @ rot_matrix
    return points

def create_stacked_chiral_layers(num_layers=8, twist_angle=15.0, z_spacing=0.1):
    """
    Create stacked 2D layers with chiral twist.
    Each layer is rotated by twist_angle relative to the previous.
    """
    layers = []
    for i in range(num_layers):
        # Base layer points
        layer_points = create_2d_layer(num_points=50, size=1.0, layer_id=i)
        # Apply cumulative twist for chirality (positive for right-handed, negative for left)
        chiral_twist = i * np.deg2rad(twist_angle)
        rot_matrix = np.array([[np.cos(chiral_twist), -np.sin(chiral_twist)], 
                               [np.sin(chiral_twist), np.cos(chiral_twist)]])
        twisted_points = layer_points @ rot_matrix
        # Add z-coordinate for stacking
        z = np.full((len(twisted_points), 1), i * z_spacing)
        stacked_points = np.hstack((twisted_points, z))
        layers.append(stacked_points)
    return layers

def interpolate_chirality(base_layers, chiral_layers, t):
    """
    Interpolate between achiral stacking (no twist) and chiral stacking.
    """
    smooth_t = t * t * (3 - 2 * t)
    interpolated_layers = []
    for i in range(len(base_layers)):
        base_layer = base_layers[i]
        chiral_layer = chiral_layers[i]
        # Interpolate x,y (twist) while keeping z fixed
        interp_xy = (1 - smooth_t) * base_layer[:, :2] + smooth_t * chiral_layer[:, :2]
        interp_layer = np.hstack((interp_xy, base_layer[:, 2:]))
        interpolated_layers.append(interp_layer)
    return interpolated_layers

# Example visualization function (add to your loop if needed)
def visualize_stacked_layers(layers, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    for layer in layers:
        ax.scatter(layer[:, 0], layer[:, 1], layer[:, 2], s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Initialize the LLM
print("="*70)
print("CHIRAL STACKED 2D LLM - TEXT GENERATION DEMO")
print("="*70)

llm = OrigamiLLM()

# Example: Create achiral and chiral stacks for potential visualization
base_twist = 0.0
chiral_twist = 15.0
base_layers = create_stacked_chiral_layers(num_layers=4, twist_angle=base_twist)
chiral_layers = create_stacked_chiral_layers(num_layers=4, twist_angle=chiral_twist)

# Interactive generation loop
print("\n" + "="*70)
print("INTERACTIVE TEXT GENERATION")
print("="*70)

while True:
    start_word = input("USER (start word): ").strip()
    if not start_word:
        continue
    
    stack_state = 1
    chirality_strength = 1  # Controls handedness bias in tuning

    print("\n--- GENERATED TEXT ---")
    text = llm.generate_text(
        start_word=start_word, 
        max_words=800,  # Reduced for demo
        stack_state=stack_state,
        chirality_strength=chirality_strength
    )
    print(text)  
    print("-" * 22 + "\n")
    
    # Optional: Visualize at interpolation t=1 (full chiral)
    # visualize_stacked_layers(interpolate_chirality(base_layers, chiral_layers, 1.0))
