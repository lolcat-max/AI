import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from collections import Counter, defaultdict
import hashlib

# ================================================================
# NEW: EMBEDDED TEMPLATE ENGINE
# ================================================================

class EmbeddedTemplateMatcher:
    """
    Finds and uses semantic templates derived from the corpus
    to guide text generation.
    """
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim
        self.templates = []  # List of (template_text, template_vector) tuples
        self.word_vectors = {}
        print("\n[Embedded Template Matcher Initialized]")
        print(f"  Vector Dimension: {self.vector_dim}")

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
        print(f"\n[Deriving {num_templates} embedded templates from corpus...]")
        # Select representative sentences as templates
        step = max(1, len(corpus) // num_templates)
        for i in range(0, len(corpus), step):
            sentence = corpus[i]
            if len(sentence.split()) > 5: # Ensure template has some substance
                template_vector = self._get_sentence_vector(sentence)
                self.templates.append((sentence, template_vector))
        print(f"  Successfully created {len(self.templates)} templates.")

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

# Simple n-gram based text generator (mimicking your linguistic binding style)
class OrigamiLLM:
    def __init__(self):
        # Training corpus about origami and neural networks
        text = open(input("Filename:"), encoding="utf-8").read()
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
    
    def generate_text(self, start_word=None, max_words=15, fold_state=0.0, tuning_strength=0.0):
        """
        Generate text based on fold state and template tuning.
        fold_state: 0.0 = unfolded (random), 1.0 = folded (coherent)
        tuning_strength: 0.0 = no tuning, 1.0 = strong template influence
        """
        if start_word is None or start_word not in self.bigrams:
            start_word = random.choice(list(self.bigrams.keys()))
        
        result = [start_word]
        current_word = start_word
        
        for _ in range(max_words - 1):
            use_model = random.random() < fold_state
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
                else:
                    break

            # --- NEW: TEMPLATE TUNING ---
            context = ' '.join(result[-5:]) # Use last 5 words as context
            best_template_tuple = self.template_matcher.find_best_template(context)
            
            probs = np.ones(len(candidates)) # Start with uniform probability
            
            if tuning_strength > 0 and best_template_tuple is not None:
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
                            # Boost is proportional to tuning strength
                            probs[boost_idx] += 10.0 * tuning_strength 
                except (ValueError, IndexError):
                    pass # Context not in template or at the end

            # Normalize probabilities
            probs /= probs.sum()
            
            # Choose next word based on potentially tuned probabilities
            next_word = np.random.choice(candidates, p=probs)
            
            result.append(next_word)
            current_word = next_word
            
            if len(result) >= 8 and next_word in ['patterns', 'space', 'understanding', 'representations', 'layers', 'model', 'network']:
                break
        
        return ' '.join(result)

# Octahedron geometry functions (unchanged)
def create_octahedron_vertices():
    return np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])

def create_octahedron_faces():
    return [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [0, 2, 5], [2, 1, 5], [1, 3, 5], [3, 0, 5],
    ]

def create_unfolded_octahedron():
    triangle_height = np.sqrt(3) / 2
    triangles = []
    
    for i in range(4):
        x_offset = i * 0.5
        y_base = 0
        if i % 2 == 0:
            triangles.append([[x_offset, y_base, 0], [x_offset + 1, y_base, 0], [x_offset + 0.5, y_base + triangle_height, 0]])
        else:
            triangles.append([[x_offset, y_base + triangle_height, 0], [x_offset + 1, y_base + triangle_height, 0], [x_offset + 0.5, y_base, 0]])
    
    for i in range(2):
        x_offset = 0.5 + i
        y_base = triangle_height
        triangles.append([[x_offset, y_base, 0], [x_offset + 1, y_base, 0], [x_offset + 0.5, y_base + triangle_height, 0]])
    
    for i in range(2):
        x_offset = 0.5 + i
        y_base = -triangle_height
        triangles.append([[x_offset, y_base, 0], [x_offset + 1, y_base, 0], [x_offset + 0.5, y_base + triangle_height, 0]])
    
    return np.array(triangles)

def interpolate_state(unfolded, folded_verts, folded_faces, t):
    smooth_t = t * t * (3 - 2 * t)
    result = []
    for i, face_indices in enumerate(folded_faces):
        if i < len(unfolded):
            unfold_tri = unfolded[i]
            fold_tri = folded_verts[face_indices]
            interp_tri = (1 - smooth_t) * unfold_tri + smooth_t * fold_tri
            result.append(interp_tri)
    return result

# Initialize the LLM
print("="*70)
print("ORIGAMI LLM - TEXT GENERATION DEMO (with Template Tuning)")
print("="*70)

llm = OrigamiLLM()

# Interactive generation loop
print("\n" + "="*70)
print("INTERACTIVE TEXT GENERATION")
print("="*70)

while True:
    start_word = input("USER (start word): ").strip()
    if not start_word:
        continue
    
    fold_state = 1
    tuning_strength = 1

    print("\n--- GENERATED TEXT ---")
    text = llm.generate_text(
        start_word=start_word, 
        max_words=800, 
        fold_state=fold_state,
        tuning_strength=tuning_strength
    )
    print(text)  
    print("-" * 22 + "\n")
