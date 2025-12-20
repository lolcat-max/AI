import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.chunk import RegexpParser

# ------------ NLTK resources ------------

for pkg in ["punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()
WH_WORDS = {"who", "what", "where", "when", "why"}

@dataclass
class PromptDescriptor:
    qtype: str
    raw: str

@dataclass
class SyntaxDescriptor:
    verb: Optional[str]
    obj: Optional[str]
    modifiers: Dict[str, str]

@dataclass
class DescriptorContext:
    qtype: str
    action: Optional[str]
    obj: Optional[str]
    modifiers: Dict[str, str]

# ------------ KB (dynamic) ------------

KNOWLEDGE_BASE: List[Dict[str, Any]] = []

def add_fact(action: str, obj: str, subject: Optional[str] = None, place: Optional[str] = None):
    fact: Dict[str, Any] = {"action": action, "obj": obj}
    if subject:
        fact["answer"] = subject   # used for WHO / WHAT
    if place:
        fact["place"] = place      # used for WHERE
    KNOWLEDGE_BASE.append(fact)


# ------------ WordNet helpers ------------

def wn_lemmatize_verb(token: str) -> str:
    v = lemmatizer.lemmatize(token, pos="v")
    if v != token:
        return v
    n = lemmatizer.lemmatize(token, pos="n")
    return n

def wn_lemmatize_noun_phrase(phrase: str) -> str:
    tokens = re.findall(r"[a-zA-Z']+", phrase.lower())
    lemmas = [lemmatizer.lemmatize(t, pos="n") for t in tokens]
    return " ".join(lemmas)

def wn_synonym_set(phrase: str) -> Set[str]:
    syns = set()
    for syn in wn.synsets(phrase.replace(" ", "_")):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            syns.add(name)
    return syns


# ------------ 0. IE: SVO + location with RegexpParser ------------

GRAMMAR = r"""
  NP: {<DT|PRP\$|JJ|NNP|NNPS|NN|NNS>+}
  VP: {<VB|VBD|VBG|VBN|VBP|VBZ><RB>*}
"""

chunker = RegexpParser(GRAMMAR)

def triples_from_sentence(sent: str):
    """
    Very shallow heuristic SVO extractor: NP before VP, NP after VP.[web:75][web:107]
    """
    tokens = word_tokenize(sent)
    tagged = pos_tag(tokens)
    tree = chunker.parse(tagged)

    subjects: List[str] = []
    verbs: List[str] = []
    objects: List[str] = []

    last_vp_seen = False

    for subtree in tree:
        if isinstance(subtree, nltk.Tree) and subtree.label() == "NP":
            np_text = " ".join(tok for tok, _ in subtree.leaves())
            if not last_vp_seen:
                subjects.append(np_text)
            else:
                objects.append(np_text)
        elif isinstance(subtree, nltk.Tree) and subtree.label() == "VP":
            vp_text = " ".join(tok for tok, _ in subtree.leaves())
            verbs.append(vp_text)
            last_vp_seen = True

    triples = []
    if subjects and verbs and objects:
        subj = subjects[0]
        obj = objects[0]
        for vp in verbs:
            v_main = vp.split()[0]
            triples.append((subj, v_main, obj))
    return triples

def location_from_sentence(sent: str):
    """
    Heuristic for 'X is in Y' / 'X is located in Y'.[web:75]
    """
    tokens = word_tokenize(sent)
    tagged = pos_tag(tokens)
    # look for pattern: NP 'is' 'located'? 'in' NP
    words = [w for w, t in tagged]
    tags = [t for w, t in tagged]

    results = []
    for i, (w, t) in enumerate(zip(words, tags)):
        if w.lower() in {"is", "was", "are"}:
            # preceding NP
            left = words[max(0, i-3):i]
            left_tags = tags[max(0, i-3):i]
            if not left:
                continue
            if not any(tag.startswith("NN") for tag in left_tags):
                continue
            thing = " ".join(left)

            # optional 'located'
            j = i + 1
            if j < len(words) and words[j].lower() == "located":
                j += 1

            # 'in' + NP
            if j < len(words) and words[j].lower() == "in":
                k = j + 1
                right = words[k:k+3]
                if right:
                    place = " ".join(right)
                    results.append((thing, "located", place))
    return results

def ingest_corpus(text: str):
    for sent in sent_tokenize(text):
        for subj, verb, obj in triples_from_sentence(sent):
            action = wn_lemmatize_verb(verb.lower())
            obj_norm = wn_lemmatize_noun_phrase(obj)
            subj_norm = wn_lemmatize_noun_phrase(subj)
            add_fact(action=action, obj=obj_norm, subject=subj_norm)

        for thing, _, place in location_from_sentence(sent):
            obj_norm = wn_lemmatize_noun_phrase(thing)
            place_norm = wn_lemmatize_noun_phrase(place)
            add_fact(action="located", obj=obj_norm, place=place_norm)


# ------------ 1. Prompt Descriptor ------------

def build_prompt_descriptor(question: str) -> PromptDescriptor:
    text = question.strip().lower()
    m = re.match(r"^(\w+)", text)
    first = m.group(1) if m else ""
    qtype = first.upper() if first in WH_WORDS else "UNKNOWN"
    return PromptDescriptor(qtype=qtype, raw=question)


# ------------ 2. Syntax Descriptor ------------

def tokenize_question(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())

def find_main_verb_q(tokens: List[str]) -> Optional[str]:
    for tok in tokens:
        if wn.synsets(tok, pos=wn.VERB):
            return tok
    return tokens[1] if len(tokens) > 1 else None

def extract_object_q(tokens: List[str], verb: Optional[str]) -> Optional[str]:
    if verb is None:
        return None
    try:
        idx = tokens.index(verb)
    except ValueError:
        return None
    stop_words = WH_WORDS | {"the", "a", "an", "of", "did", "do", "is", "was", "in"}
    obj_tokens = [t for t in tokens[idx + 1 :] if t not in stop_words]
    if not obj_tokens:
        return None
    return " ".join(obj_tokens)

def build_syntax_descriptor(pd: PromptDescriptor) -> SyntaxDescriptor:
    tokens = tokenize_question(pd.raw)
    verb = find_main_verb_q(tokens)
    obj = extract_object_q(tokens, verb)
    modifiers: Dict[str, str] = {}
    if "when" in tokens:
        modifiers["ASK_TIME"] = "true"
    if "where" in tokens:
        modifiers["ASK_PLACE"] = "true"
    return SyntaxDescriptor(verb=verb, obj=obj, modifiers=modifiers)


# ------------ 3. Descriptor Context ------------
def build_context(pd: PromptDescriptor, sd: SyntaxDescriptor) -> DescriptorContext:
    action = wn_lemmatize_verb(sd.verb) if sd.verb else None
    obj = wn_lemmatize_noun_phrase(sd.obj) if sd.obj else None
    return DescriptorContext(qtype=pd.qtype, action=action, obj=obj, modifiers=sd.modifiers)


# ------------ 4. Inference ------------
def object_matches(kb_obj: str, query_obj: Optional[str]) -> bool:
    if not query_obj:
        return False
    if kb_obj == query_obj:
        return True
    q_syns = wn_synonym_set(query_obj)
    kb_syns = wn_synonym_set(kb_obj)
    if kb_obj in q_syns or query_obj in kb_syns:
        return True
    if q_syns.intersection(kb_syns):
        return True
    return False


def infer_answer(ctx: DescriptorContext) -> Dict[str, Any]:
    for fact in KNOWLEDGE_BASE:
        kb_obj = fact.get("obj")
        kb_action = fact.get("action")
        if kb_obj is None or kb_action is None:
            continue

        # WHO/WHERE rely mainly on object semantics
        if ctx.qtype in {"WHO", "WHERE"}:
            if not object_matches(kb_obj, ctx.obj):
                continue
        else:
            if ctx.action and kb_action != ctx.action:
                continue
            if not object_matches(kb_obj, ctx.obj):
                continue

        if ctx.qtype == "WHO" and "answer" in fact:
            return {"person": fact["answer"], "obj": kb_obj, "action": kb_action}

        if ctx.qtype == "WHERE" and "place" in fact:
            return {"place": fact["place"], "obj": kb_obj, "action": kb_action}

        if ctx.qtype == "WHAT" and "answer" in fact:
            return {"thing": f"{kb_obj} is associated with {fact['answer']}"}

    return {}


# ------------ 5. Realization ------------

def realize_answer(question: str, ctx: DescriptorContext, bindings: Dict[str, Any]) -> str:
    if not bindings:
        return "I do not know the answer to that."
    obj = bindings.get("obj", ctx.obj)
    action = bindings.get("action", ctx.action)
    if "person" in bindings:
        return f"{bindings['person']} {action} {obj}."
    if "place" in bindings:
        return f"{obj.capitalize()} is located in {bindings['place']}."
    if "thing" in bindings:
        return bindings["thing"]
    return "I have some data, but cannot express it clearly."


# ------------ Public API ------------

def answer_question(question: str) -> str:
    pd = build_prompt_descriptor(question)
    sd = build_syntax_descriptor(pd)
    ctx = build_context(pd, sd)
    bindings = infer_answer(ctx)
    return realize_answer(question, ctx, bindings)


# ------------ Demo ------------

if __name__ == "__main__":
    try:
        with open(input("Filename: "), "r", encoding="utf-8") as f: corpus = f.read().lower()
    except:
        try:
            corpus = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text.lower()
        except:
            corpus = "hello world " * 1000

    ingest_corpus(corpus)
    while True:
        print("A:", answer_question(input("USER: ")))
        print()
