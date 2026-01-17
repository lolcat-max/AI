#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynthReason (Full NLTK Semantic KB + Causal Memory + Fixed Response Logic)
Handles "what do people care?", "why people work?", etc. without any *.txt files.

One-time setup:
    pip install nltk
    python -c "import nltk; nltk.download('punkt averaged_perceptron_tagger maxent_ne_chunker words wordnet omw-1.4')"
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import wordnet as wn


@dataclass
class Config:
    """Descriptor labels (customizable)."""
    desname: str = "who"
    desobject: str = "what"
    desaction: str = "how"
    destime: str = "when"
    desdescription: str = "describe"
    desnumber: str = "count"
    desreason: str = "why"
    desplace: str = "where"
    desemotion: str = "feel"


@dataclass
class Memory:
    """Causal memory: (subj_lemma, verb_lemma) → reason_phrase."""
    why_map: Dict[Tuple[str, str], str] = field(default_factory=dict)


class NltkSemanticKB:
    """
    Categorize tokens using NLTK POS + WordNet lexnames + NER.
    Replaces all *.txt files (objects.txt, places.txt, reason.txt, etc.). [web:79][web:65][web:68]
    """

    # POS → descriptor (covers ~95% of tokens reliably) [web:79][web:81]
    POS_TO_DESC = {
        # People/Names
        "NNP": "who",   # Proper noun (John, USA)
        "NNPS": "who",  # Proper plural (Americans)
        "PRP": "who",   # Pronouns (I, you, they)
        # Places/Nouns (collect nouns for "what")
        "NN": "where",  # Common noun (city, house, money)
        "NNS": "where", # Plural nouns (cities, people)
        # Actions/How
        "VB": "how",    # Verb base (run)
        "VBD": "how",   # Past (ran)
        "VBG": "how",   # Gerund (running)
        "VBN": "how",   # Past participle (run)
        "VBP": "how",   # Present non-3sg (run, do)
        "VBZ": "how",   # Present 3sg (runs)
        # Descriptions/Adjectives
        "JJ": "describe",   # Adjective (big, intelligent)
        "JJR": "describe",  # Comparative (bigger)
        "JJS": "describe",  # Superlative (biggest)
        "RB": "describe",   # Adverb (quickly, yes)
        "RBR": "describe",  # Comparative adverb
        "RBS": "describe",  # Superlative adverb
        # Numbers
        "CD": "count",      # Cardinal (one, 42%)
        # Time (heuristic)
        "IN": "when",       # Prepositions (in 2024, at noon)
        # Fallback
        "DT": "what",       # Determiners (the, a, what)
    }

    # WordNet lexname overrides (tiebreaker for ambiguous POS) [web:65][web:66]
    LEXNAME_OVERRIDES = {
        "noun.time": "when",
        "noun.motive": "why",
        "noun.possession": "why", 
        "noun.feeling": "feel",
        "noun.act": "how",
    }

    # NER mappings [web:68][web:76]
    NE_TO_DESC = {
        "PERSON": "who",
        "GPE": "where",
        "LOCATION": "where",
        "DATE": "when",
        "TIME": "when",
        "MONEY": "why",
        "ORGANIZATION": "who",
    }

    @staticmethod
    def _pos_to_wn(pos_tag: str) -> Optional[str]:
        if pos_tag.startswith("NN"):
            return wn.NOUN
        if pos_tag.startswith("VB"):
            return wn.VERB
        if pos_tag.startswith("JJ"):
            return wn.ADJ
        if pos_tag.startswith("RB"):
            return wn.ADV
        return None

    def _wordnet_lexname_vote(self, word: str, wn_pos: Optional[str]) -> Optional[str]:
        synsets = wn.synsets(word, pos=wn_pos) if wn_pos else wn.synsets(word)
        if not synsets:
            return None
        lex = synsets[0].lexname()  # First sense [web:65]
        return self.LEXNAME_OVERRIDES.get(lex)

    def descriptor_for_token(self, token: str, pos_tag_str: str, ne_label: Optional[str]) -> str:
        # 1. NER (contextual)
        if ne_label and ne_label in self.NE_TO_DESC:
            return self.NE_TO_DESC[ne_label]

        # 2. POS tag (reliable primary) [web:79]
        desc = self.POS_TO_DESC.get(pos_tag_str, "what")
        if desc != "what":
            return desc

        # 3. WordNet tiebreaker
        wn_pos = self._pos_to_wn(pos_tag_str)
        wn_lex = self._wordnet_lexname_vote(token.lower(), wn_pos)
        if wn_lex:
            return wn_lex

        return "what"


def extract_ne_labels(tokens: List[str]) -> Dict[str, str]:
    """Token → NE label (e.g., {'money': 'MONEY'}). [web:68]"""
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)
    out = {}
    for node in tree:
        if hasattr(node, "label"):
            label = node.label()
            for leaf_word, _leaf_pos in node.leaves():
                out[leaf_word] = label
    return out


def normalize_lemma(token: str) -> str:
    """Simple lowercase lemma."""
    return token.lower()


def extract_subject_verb(tokens: List[str], tags: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """Find (subject_noun, verb) pair."""
    verb_i = None
    for i, (_w, t) in enumerate(tags):
        if t.startswith("VB"):
            verb_i = i
            break
    if verb_i is None:
        return None

    subj = None
    for j in range(verb_i - 1, -1, -1):
        w, t = tags[j]
        if t.startswith("NN"):
            subj = w
            break

    if not subj:
        return None

    verb = tags[verb_i][0]
    return (normalize_lemma(subj), normalize_lemma(verb))


def learn_because(memory: Memory, user_input: str):
    """Parse/learn: "<subj> <verb> ... because (of) <reason>"."""
    s = user_input.strip()
    m = re.search(r"\bbecause\b(?:\s+of)?\s+(.*)$", s, flags=re.IGNORECASE)
    if not m:
        return

    reason = m.group(1).strip().rstrip(".?!")
    tokens = word_tokenize(s)
    tags = pos_tag(tokens)
    sv = extract_subject_verb(tokens, tags)
    if not sv:
        return

    memory.why_map[sv] = reason


def answer_why(memory: Memory, question: str) -> Optional[str]:
    """Retrieve learned causal answer for "why <subj> <verb>?"."""
    tokens = word_tokenize(question)
    tags = pos_tag(tokens)
    sv = extract_subject_verb(tokens, tags)
    if not sv:
        return None
    return memory.why_map.get(sv)


def process_language_nltk(user_input: str, cfg: Config, kb: NltkSemanticKB) -> str:
    """Build [desc>word:desc>word:...] using NLTK categorisation."""
    tokens = word_tokenize(user_input)
    tags = pos_tag(tokens)
    ne_map = extract_ne_labels(tokens)

    segs = ["["]
    for w, t in tags:
        if re.fullmatch(r"[.,!?;:()]", w):
            continue
        ne_label = ne_map.get(w)
        desc = kb.descriptor_for_token(w, t, ne_label)
        segs.append(f"{desc}>{w}:")
    segs.append("]")
    return "".join(segs)


def process_response(answerstring: str, user_input: str, cfg: Config, memory: Memory) -> str:
    """1. Causal memory, 2. Context-aware descriptor extraction."""
    # 1. "why" from memory
    if re.search(r"^\s*why\b", user_input.strip(), flags=re.IGNORECASE):
        learned = answer_why(memory, user_input)
        if learned:
            return learned

    # Query type
    q_lower = user_input.lower()
    if "why" in q_lower:
        key = "why"
    elif "where" in q_lower:
        key = "where"
    elif "when" in q_lower:
        key = "when"
    elif "who" in q_lower:
        key = "who"
    elif "how" in q_lower:
        key = "how"
    else:
        key = "what"

    # Skip query words ("what", "do", "why", etc.)
    skip_words = {"what", "do", "does", "did", "why", "where", "when", "who", "how", "yes", "a", "an", "the"}
    core = answerstring.strip().lstrip("[").rstrip("]")
    picks = []

    for segment in core.split(":"):
        if not segment or ">" not in segment:
            continue
        desc, word = segment.split(">", 1)
        word = word.strip()

        # Skip query words
        if word.lower() in skip_words:
            continue

        # Match descriptor (flexible for "what")
        if key == "what":
            # "what" → nouns (where/who) + verbs (how)
            if desc in ("where", "who", "how"):
                picks.append(word)
        elif desc == key:
            picks.append(word)

    return " ".join(picks[:3]) if picks else "I don't know yet."


def process_knowledge(user_input: str, cfg: Config, kb: NltkSemanticKB, memory: Memory) -> str:
    """Full pipeline."""
    learn_because(memory, user_input)
    answerstring = process_language_nltk(user_input, cfg, kb)
    return process_response(answerstring, user_input, cfg, memory)


def main():
    cfg = Config()
    kb = NltkSemanticKB()
    memory = Memory()

    print("=== SynthReason (NLTK Semantic KB + Causal Memory) ===")
    print("Examples:")
    print("  people work because of money.")
    print("  why people work?")
    print("  what do people care?")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"quit", "exit", ""}:
            break

        response = process_knowledge(user_input, cfg, kb, memory)
        print("AI:", response, "\n")


if __name__ == "__main__":
    main()
