"""
Enhanced Transitioning for Streaming Text Generation
Implements smooth token-to-token transitions with n-gram bindings
NOW WITH VECTOR SYMBOLIC GRAPH NODE PREDICTION
"""
KB_LEN = -1
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import pickle
import os

# =====================================================================
# VECTOR SYMBOLIC ARCHITECTURE
# =====================================================================
class VectorSymbolicArchitecture:
    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.codebook: Dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    def create_vector(self, normalize: bool = True) -> np.ndarray:
        vec = np.random.randn(self.dimensions)
        if normalize:
            vec = vec / np.linalg.norm(vec)
        return vec

    def bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        result = np.fft.ifft(fft_a * fft_b)
        return np.real(result)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        return np.mean(vectors, axis=0)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def add_to_codebook(self, symbol: str) -> np.ndarray:
        with self.lock:
            if symbol not in self.codebook:
                self.codebook[symbol] = self.create_vector()
            return self.codebook[symbol]

    def save_codebook(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.codebook, f)
        print(f"✓ Codebook saved to {filepath}")

    def load_codebook(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.codebook = pickle.load(f)
        print(f"✓ Codebook loaded from {filepath}")


# =====================================================================
# VSA GRAPH (replaces TransitionEncoder)
# =====================================================================
class VSAGraph:
    """
    Represents the corpus as a graph where tokens are nodes and
    n-gram transitions are directed, weighted edges, encoded using VSA.
    """
    def __init__(self, vsa):
        self.vsa = vsa
        self.bigram_edges = {}
        self.trigram_edges = {}
        self.lock = threading.Lock()

    def _add_bigram_edge(self, token1: str, token2: str) -> np.ndarray:
        key = (token1, token2)
        with self.lock:
            if key not in self.bigram_edges:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                self.bigram_edges[key] = self.vsa.bind(vec1, vec2)
            return self.bigram_edges[key]

    def _add_trigram_edge(self, token1: str, token2: str, token3: str) -> np.ndarray:
        key = (token1, token2, token3)
        with self.lock:
            if key not in self.trigram_edges:
                vec1 = self.vsa.add_to_codebook(token1)
                vec2 = self.vsa.add_to_codebook(token2)
                vec3 = self.vsa.add_to_codebook(token3)
                bound12 = self.vsa.bind(vec1, vec2)
                self.trigram_edges[key] = self.vsa.bind(bound12, vec3)
            return self.trigram_edges[key]

    def _process_path_batch(self, paths: List[List[str]]) -> Tuple[int, int]:
        bigram_count = 0
        trigram_count = 0
        for path in paths:
            for i in range(len(path) - 1):
                self._add_bigram_edge(path[i], path[i+1])
                bigram_count += 1
            for i in range(len(path) - 2):
                self._add_trigram_edge(path[i], path[i+1], path[i+2])
                trigram_count += 1
        return bigram_count, trigram_count

    def learn_graph_from_corpus(self, corpus: List[List[str]], max_workers: int = 8, batch_size: int = 100):
        print("Learning graph structure from corpus...")
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_path_batch, batch) for batch in batches}
            with tqdm(total=len(batches), desc="Learning Edges", ncols=80) as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)
        print(f"  ✓ Graph built with {len(self.bigram_edges)} unique bigram edges.")
        print(f"  ✓ Graph built with {len(self.trigram_edges)} unique trigram edges.")

    def predict_next_nodes(self, context_path: List[str]) -> List[Tuple[str, float]]:
        """Predicts potential next nodes given a context path."""
        if not context_path:
            return []
        
        candidates = defaultdict(float)
        last_node = context_path[-1]
        for (node1, node2), _ in self.bigram_edges.items():
            if node1 == last_node:
                candidates[node2] += 1.0

        if len(context_path) >= 2:
            last_two_nodes = tuple(context_path[-2:])
            for (node1, node2, node3), _ in self.trigram_edges.items():
                if (node1, node2) == last_two_nodes:
                    candidates[node3] += 1.5

        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    def save_model(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bigram_edges.pkl"), 'wb') as f:
            pickle.dump(self.bigram_edges, f)
        with open(os.path.join(directory, "trigram_edges.pkl"), 'wb') as f:
            pickle.dump(self.trigram_edges, f)
        print(f"✓ Graph model saved in {directory}")

    def load_model(self, directory: str):
        with open(os.path.join(directory, "bigram_edges.pkl"), 'rb') as f:
            self.bigram_edges = pickle.load(f)
        with open(os.path.join(directory, "trigram_edges.pkl"), 'rb') as f:
            self.trigram_edges = pickle.load(f)
        print(f"✓ Graph model loaded from {directory}")

# =====================================================================
# GRAPH NODE PREDICTOR (replaces AdaptiveTransitionGenerator)
# =====================================================================
class GraphNodePredictor:
    """
    Generates sequences by traversing the VSAGraph, predicting the next node at each step.
    """
    def __init__(self, vsa, vsa_graph):
        self.vsa = vsa
        self.vsa_graph = vsa_graph

    def generate_path(self, seed_path: List[str], max_len: int = 50, exploration: float = 0.3):
        current_path = seed_path.copy()
        for _ in range(max_len):
            candidate_nodes = self.vsa_graph.predict_next_nodes(current_path)
            if not candidate_nodes:
                break
            
            if np.random.random() < exploration:
                idx = min(len(candidate_nodes) - 1, int(np.random.exponential(1.5)))
                next_node = candidate_nodes[idx][0]
            else:
                next_node = candidate_nodes[0][0]
            
            yield next_node
            current_path.append(next_node)
            # Online learning: dynamically add the traversed edge
            if len(current_path) >= 2:
                self.vsa_graph._add_bigram_edge(current_path[-2], current_path[-1])

# =====================================================================
# DEMONSTRATION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("="*80)
    print("VECTOR SYMBOLIC GRAPH NODE PREDICTION FOR TEXT GENERATION")
    print("="*80)

    vsa = VectorSymbolicArchitecture(dimensions=2048)
    vsa_graph = VSAGraph(vsa)
    model_dir = "vsa_graph_model"

    
    # --- Training Phase ---
    print("\n[1] TRAINING: Building Graph from Corpus")
    print("-"*80)
    filename = input("Filename: ")
    with open(filename, encoding="utf-8") as f:
        raw_text = f.read()[:KB_LEN]

    sentences = raw_text.strip().split("\n")
    corpus = [s.split() for s in sentences if s]
    print(f"Corpus loaded: {len(corpus)} paths (sequences)")

    vsa_graph.learn_graph_from_corpus(corpus, max_workers=4, batch_size=10)

    print("\nBuilding node vocabulary...")
    for path in tqdm(corpus, desc="Vocabulary", ncols=80):
        for token_node in path:
            vsa.add_to_codebook(token_node)
    print(f"  ✓ Vocabulary size: {len(vsa.codebook)} unique nodes")

    vsa.save_codebook(os.path.join(model_dir, "codebook.pkl"))
    vsa_graph.save_model(model_dir)

    # --- Generation Phase ---
    print("\n[2] PREDICTION: Generating Text by Traversing the Graph")
    print("-"*80)
    predictor = GraphNodePredictor(vsa, vsa_graph)
    while True:
        seed_text = input("USER: ")
        print(f"SEED PATH: '{seed_text}'")
        
        print("GENERATED PATH: ", end='')
        output_path = []
        for token in predictor.generate_path(
            seed_text.split(),
            max_len=800,
            exploration=0.8
        ):
            output_path.append(token)
            print(token, end=' ', flush=True)
        print("\n\nDemonstration complete.")
