import random

def get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def build_ngram_transitions(words, n=2):
    transitions = {}
    for gram in get_ngrams(words, n):
        prefix = gram[:-1]
        next_word = gram[-1]
        transitions.setdefault(prefix, []).append(next_word)
    return transitions

def get_ngrams_set(words, n):
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

def ai_inference(q1_words, q2_words, n=2):
    a1 = f"answer for '{q1_words[0]}'"
    a2 = f"answer for '{q2_words[0]}'"

    ngrams_q1 = get_ngrams_set(q1_words, n)
    ngrams_q2 = get_ngrams_set(q2_words, n)

    duplicate_ngrams = ngrams_q1.intersection(ngrams_q2)
    return len(duplicate_ngrams) == 0 and a1 == a2

def generate_text_guided(transitions, n=2, length=50, window_size=4):
    current_prefix = random.choice(list(transitions.keys()))
    output_words = list(current_prefix)

    # Keep a history list of recent windows for inference checks
    recent_windows = [output_words[-window_size:]]

    for _ in range(length - (n - 1)):
        next_words = transitions.get(current_prefix)
        if not next_words:
            break
        
        # Filter next words using inference pairs
        viable_next_words = []
        for w in next_words:
            candidate_seq = output_words[-(window_size - 1):] + [w]
            # Check inference against recent windows
            inference_triggered = False
            for recent in recent_windows:
                if ai_inference(recent, candidate_seq, n):
                    inference_triggered = True
                    break
            if not inference_triggered:
                viable_next_words.append(w)
        
        # If no viable word (all trigger inference), fallback to all options anyway
        if not viable_next_words:
            viable_next_words = next_words
        
        next_word = random.choice(viable_next_words)
        output_words.append(next_word)
        current_prefix = tuple(output_words[-(n-1):])
        # Update recent windows
        if len(output_words) >= window_size:
            recent_windows.append(output_words[-window_size:])
            # Optionally limit history size to save memory
            if len(recent_windows) > 10:
                recent_windows.pop(0)

    return output_words

def read_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    for ch in ['?', '.', ',', '\n', '!', ':', ';']:
        text = text.replace(ch, ' ')
    words = text.split()
    return words

# Usage example:
words = read_text_file('xaa')
transitions = build_ngram_transitions(words, n=2)
generated = generate_text_guided(transitions, n=2, length=500, window_size=8)
print(' '.join(generated))
