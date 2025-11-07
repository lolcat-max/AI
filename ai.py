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

# Simple n-gram based text generator (mimicking your linguistic binding style)
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
        
        # NEW: Self-judgment components
        self.rules = []  # List of generated rules (strings like "Ignore words shorter than 4 letters")
        self.rule_explanations = {}  # Rule name to explanation

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

    def generate_rules(self, num_rules=5):
        """
        NEW: Generate rules to follow during text generation.
        Rules are philosophical: e.g., assign significance contextually, dismiss insignificant details.
        Uses n-gram model to create rule-like sentences, parsed into actionable rules.
        """
        self.rules = []
        self.rule_explanations = {}
        
        rule_starters = [
            "ignore short words", "prioritize coherent phrases", "dismiss punctuation as insignificant",
            "elevate key terms like 'network' or 'fold'", "reverse dismissal for reversal context",
            "assign significance to words longer than", "use templates for rule binding",
            "judge coherence by repetition limit"
        ]
        
        for i in range(num_rules):
            # Generate a rule phrase using n-gram (start from a starter)
            start = random.choice(rule_starters)
            rule_phrase = self.generate_text(start_word=start.split()[-1] if start.split() else "ignore", max_words=10)
            
            # Parse into a rule (simple regex/keyword extraction for actionability)
            if "ignore" in rule_phrase.lower():
                rule = "Ignore words shorter than 4 letters"
                explanation = "Short words deemed insignificant for compression-like generation."
            elif "prioritize" in rule_phrase.lower():
                rule = "Prioritize coherent phrases from templates"
                explanation = "Contextual significance assigned to template-matched sequences."
            elif "dismiss" in rule_phrase.lower():
                rule = "Dismiss punctuation and spaces"
                explanation = "Insignificant details dismissed to focus on core message."
            elif "elevate" in rule_phrase.lower():
                rule = "Elevate key domain terms (e.g., 'network', 'fold')"
                explanation = "Rule assigns higher significance to topic-relevant words."
            elif "reverse" in rule_phrase.lower():
                rule = "Reverse dismissal if output lacks variety"
                explanation = "Every dismissal invites potential reversal for balance."
            else:
                rule = "General coherence: Limit repetition to 3 occurrences"
                explanation = "Judge for balanced significance across elements."
            
            self.rules.append(rule)
            self.rule_explanations[rule] = explanation

    def judge_output(self, generated_text: str, rules: list = None) -> dict:
        """
        NEW: Self-judge generated text against rules.
        Returns judgment score (0-1), violations, and suggestions (e.g., reverse a dismissal).
        Ties to philosophy: Checks contextual assignment of significance, reversibility.
        """
        if rules is None:
            rules = self.rules
        
        judgments = {}
        score = 1.0
        violations = []
        suggestions = []
        
        words = generated_text.split()
        word_count = len(words)
        short_words = sum(1 for w in words if len(w) <= 3)
        unique_words = len(set(words))
        repetitions = max(Counter(words).values()) if words else 0
        
        for rule in rules:
            compliance = 1.0
            if "shorter than 4" in rule:
                ratio_short = short_words / word_count if word_count else 0
                compliance = 1 - ratio_short  # Higher compliance if fewer short words ignored properly
                if ratio_short > 0.3:
                    violations.append(f"Too many short words ignored ({ratio_short:.2f}); significance under-assigned.")
                    suggestions.append("Reverse: Include more short words for variety.")
            elif "coherent phrases" in rule:
                # Simple coherence: avg word length, unique ratio
                coherence = (np.mean([len(w) for w in words]) / 5) * (unique_words / word_count)
                compliance = min(coherence, 1.0)
                if compliance < 0.5:
                    violations.append("Low coherence; templates not followed.")
                    suggestions.append("Reverse: Boost template tuning.")
            elif "punctuation" in rule:
                punct_ratio = len(re.findall(r'[.,!?;:\s]', generated_text)) / len(generated_text)
                compliance = 1 - (punct_ratio - 0.1) if punct_ratio > 0.1 else 1.0  # Dismissal should reduce punct
                if punct_ratio > 0.2:
                    violations.append("Excess punctuation not dismissed.")
                    suggestions.append("Reverse: Strip more insignificant chars.")
            elif "key terms" in rule:
                key_terms = ['network', 'fold', 'layer', 'model', 'space', 'pattern']
                key_usage = sum(1 for term in key_terms if term in generated_text.lower())
                compliance = min(key_usage / 2, 1.0)  # Expect at least 2 keys
                if key_usage < 1:
                    violations.append("Key terms not elevated to significance.")
                    suggestions.append("Reverse: Force inclusion of domain terms.")
            elif "repetition" in rule:
                compliance = 1 - (repetitions - 1) / 10 if repetitions > 1 else 1.0
                if repetitions > 3:
                    violations.append(f"High repetition ({repetitions}); balance lost.")
                    suggestions.append("Reverse: Diversify word choices.")
            else:  # General
                compliance = unique_words / word_count if word_count else 0
            
            judgments[rule] = compliance
            score *= compliance
        
        score = max(0, min(1, score ** (1/(len(rules)+1))))  # Geometric mean for overall score
        
        return {
            "overall_score": score,
            "judgments": judgments,
            "violations": violations,
            "suggestions": suggestions,
            "recommend_reversal": len(violations) > 1  # If many violations, suggest reversal
        }

    def generate_text(self, start_word=None, max_words=15, fold_state=0.0, tuning_strength=0.0):
        """
        Generate text based on fold state and template tuning.
        NOW: Generates rules first, generates text following them, then self-judges.
        """
        # NEW: Generate rules before generation
        
        if start_word is None or start_word not in self.bigrams:
            start_word = random.choice(list(self.bigrams.keys()))
        
        result = [start_word]
        current_word = start_word
        
        # Apply rules during generation (simple enforcement)
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
               
            # --- TEMPLATE TUNING (as before) ---
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

            # NEW: Rule enforcement during selection
            # E.g., filter candidates to follow rules (prioritize long words, key terms)
            filtered_candidates = []
            for cand in candidates:
                # Rule: Ignore short words (filter out if <4 letters)
                if any("shorter than 4" in r for r in self.rules) and len(cand) <= 3:
                    continue
                # Rule: Elevate key terms (boost if matches)
                if any("key terms" in r for r in self.rules) and any(term in cand for term in ['network', 'fold', 'layer']):
                    filtered_candidates.insert(0, cand)  # Prioritize
                else:
                    filtered_candidates.append(cand)
            
            if filtered_candidates:
                candidates = filtered_candidates
                probs = np.ones(len(candidates))  # Reset probs for filtered

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
        
        # NEW: Self-judge after generation
        judgment = self.judge_output(generated)
       
        
        return generated

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
print("ORIGAMI LLM - TEXT GENERATION DEMO (with Rule Generation & Self-Judgment)")
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
    llm.generate_rules(num_rules=50)

    print("\n--- GENERATED TEXT ---")
    text = llm.generate_text(
        start_word=start_word, 
        max_words=800,  # Reduced for demo
        fold_state=fold_state,
        tuning_strength=tuning_strength
    )
    print(text)  
    print("-" * 22 + "\n")
