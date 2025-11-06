import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

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
    
    def build_model(self):
        """Build n-gram probability model"""
        for sentence in self.corpus:
            words = sentence.split()
            
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
    
    def generate_text(self, start_word=None, max_words=15, fold_state=0.0):
        """
        Generate text based on fold state
        fold_state: 0.0 = unfolded (random), 1.0 = folded (coherent)
        """
        if start_word is None:
            start_word = random.choice(list(self.bigrams.keys()))
        
        result = [start_word]
        current_word = start_word
        
        for _ in range(max_words - 1):
            # Use fold_state to blend between random and learned patterns
            use_model = random.random() < fold_state
            
            if use_model and current_word in self.bigrams:
                # Use learned n-gram
                if len(result) >= 2 and (result[-2], result[-1]) in self.trigrams:
                    # Try trigram first (more coherent)
                    next_word = random.choice(self.trigrams[(result[-2], result[-1])])
                else:
                    # Fall back to bigram
                    next_word = random.choice(self.bigrams[current_word])
            else:
                # Random selection (unfolded state)
                if self.bigrams:
                    next_word = random.choice(list(self.bigrams.keys()))
                else:
                    break
            
            result.append(next_word)
            current_word = next_word
            
            # Stop if we hit a good ending
            if len(result) >= 8 and next_word in ['patterns', 'space', 'understanding', 'representations', 'layers']:
                break
        
        return ' '.join(result)

# Octahedron geometry functions
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
            triangles.append([[x_offset, y_base, 0], [x_offset + 1, y_base, 0], 
                            [x_offset + 0.5, y_base + triangle_height, 0]])
        else:
            triangles.append([[x_offset, y_base + triangle_height, 0], 
                            [x_offset + 1, y_base + triangle_height, 0],
                            [x_offset + 0.5, y_base, 0]])
    
    for i in range(2):
        x_offset = 0.5 + i
        y_base = triangle_height
        triangles.append([[x_offset, y_base, 0], [x_offset + 1, y_base, 0],
                        [x_offset + 0.5, y_base + triangle_height, 0]])
    
    for i in range(2):
        x_offset = 0.5 + i
        y_base = -triangle_height
        triangles.append([[x_offset, y_base, 0], [x_offset + 1, y_base, 0],
                        [x_offset + 0.5, y_base + triangle_height, 0]])
    
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
print("ORIGAMI LLM - TEXT GENERATION DEMO")
print("="*70)

llm = OrigamiLLM()

# Generate text at different fold states
print("="*70)
print("TEXT GENERATION AT DIFFERENT FOLD STATES")
print("="*70)

fold_states = [0.0, 0.25, 0.5, 0.75, 1.0]

while True:
    text = llm.generate_text(start_word=input("USER: "), max_words=800, fold_state=41.0)
    print(text)  
    print()
    
