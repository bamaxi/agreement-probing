import typing as T
from pathlib import Path

import datasets
from datasets import load_dataset
import spacy
import spacy.displacy
from spacy.matcher import DependencyMatcher
import spacy.tokens
from tqdm.auto import tqdm

import pandas as pd


RANDOM_SEED = 42

_PROJECT_PATH = "./wikipedia-data/"
PROJECT_PATH = Path(_PROJECT_PATH)
RESULTS_PATH = PROJECT_PATH / "results/"


MODEL_SIZE = "md"
# the date must have dumpstatus.json
WIKIPEDIA_DATE = "20240401"  # "20231220"


CUSTOM_BOUNDARIES = {" " * 2, "\n", "\n\n"}


@spacy.language.Language.component("sentence_segmenter")
def set_custom_boundaries(doc: spacy.tokens.Doc):
    """Add sentence boundaries based on newline characters. Helps further pipeline."""
    for token in doc[:-1]:
        if token.text in CUSTOM_BOUNDARIES or any(b in token.text for b in CUSTOM_BOUNDARIES):
            doc[token.i+1].is_sent_start = True
    return doc


def get_models():
    """Create spacy model for annotation and sentence finder"""
    nlp = spacy.load(f"ru_core_news_{MODEL_SIZE}")
    nlp.add_pipe("sentence_segmenter", first=True)
    
    matcher = DependencyMatcher(nlp.vocab)
    pattern = [
        {"RIGHT_ID": "root_id",
         "RIGHT_ATTRS": {"DEP": "ROOT",
                         "POS": {"IN": ["ADJ", "VERB", "AUX"]},}
        },
        {
        "LEFT_ID": "root_id",
        "REL_OP": ">--",        # root is parent and follows subject 
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"REGEX": "nsubj*"}, "POS": {"IN": ["NOUN"]},
                        "MORPH": {"IS_SUPERSET": ["Case=Nom"]}}
        },
        {
        "LEFT_ID": "subject",
        "REL_OP": ">++",        # subject is parent and precedes distractor
        "RIGHT_ID": "distractor",
        "RIGHT_ATTRS": {"DEP": "nmod", "POS": {"IN": ["NOUN"]}, "MORPH": {"IS_SUPERSET": ["Case=Acc"]}}
        }
    ]
    matcher.add("FOUNDED", [pattern])
    return nlp, matcher


def get_dataset():
    dataset = load_dataset("wikimedia/wikipedia", language="ru", date=WIKIPEDIA_DATE)
    texts = dataset["train"]["text"][100000:500000]
    return texts


def get_dataset2(skip=100000, take=500000-100000, num_proc=16):
    """Downloads the olm/wikipedia dump for the specified date"""
    max_take_i = skip + take

    dataset = load_dataset(
        "olm/wikipedia", language="ru", date=WIKIPEDIA_DATE,
        split="train",
        # streaming=True,
        trust_remote_code=True,
        num_proc=num_proc
    )
    train = dataset
    print(vars(train))
    items = train.skip(skip)
    # items = train

    def texts_generator():
        for i, item in enumerate(items):
            if i >= max_take_i:
                break

            yield item["text"]

    return texts_generator()


def download_dataset(num_proc=16):
    """Downloads the olm/wikipedia dump for the specified date"""
    dataset = load_dataset(
        "olm/wikipedia", language="ru", date=WIKIPEDIA_DATE,
        split="train",
        # streaming=True,
        trust_remote_code=True,
        num_proc=num_proc
    )
    return dataset


def iterate_dataset(dataset: datasets.Dataset, shuffle=False, skip=100000, take=500000-100000):
    """Iterates dataset, skipping first `skip` sentences and taking next `take` sentences"""
    if shuffle:
        dataset = dataset.shuffle(RANDOM_SEED)

    max_take_i = skip + take
    def texts_generator():
        for i, item in enumerate(dataset):
            if i >= max_take_i:
                break

            yield item["text"]

    return texts_generator()


def get_words_after_root(
    sentence: spacy.tokens.Span, keep_clausal: bool = False
) -> T.List[spacy.tokens.Token]:
    """Select minimally the words after root, as few as possible
    
    Skips conjunct clauses"""
    words = [sentence[0]]

    for word in sentence[1:]:
        if (
            (word.dep_=="conj" and word.pos_=="VERB")
            # conjunction of a clause
            # TODO: should we check the pos or ban anything that is a child to `*cl*`?
            or (not keep_clausal and ("CONJ" in word.pos_ and "cl" in word.head.dep_))     
            or (word.head.dep_=="conj" and word.head.pos_== "VERB" and word.head.dep_!="ROOT")
            # not a verb but has subject
            or (word.head.dep_ == "conj" and any(tok.dep_ == "nsubj" for tok in word.head.children))
            # extraneous noun modifiers
            or (word.dep_=="nmod" and word.morph.get("Case")!=['Gen'])
            or (word.head.dep_=="nmod" and word.head.morph.get("Case")!=["Gen"])
        ):
            break
        else:
            words.append(word)
    return words


def shorten_sentence(
    sentence: spacy.tokens.Span, subject_id: int, distractor_id: int, root_id: int,
    keep_clausal: bool = False
) -> T.List[spacy.tokens.Token]:
    """Shorten the sentence to isolate agreement at the beggining of the new short sentence"""

    before_subject = [word for word in sentence[:subject_id] if sentence[subject_id].is_ancestor(word)]
    subj_to_distr = list(sentence[subject_id:distractor_id+1])
    dist_to_root = [word for word in sentence[distractor_id:root_id]
                    if (word.head.dep_=="advmod" or word.dep_ in {"advmod", "aux:pass", "cop"})]
    after_root = get_words_after_root(sentence[root_id:], keep_clausal=keep_clausal)

    sent = before_subject + subj_to_distr + dist_to_root + after_root
    return sent


def get_full_predicate(
    sentence: spacy.tokens.Span, subject_id: int, distractor_id: int, root_id: int
):
    """Fetch full predicate (with auxillary verbs)"""
    root = sentence[root_id]
    aux = [tok for tok in sentence if (tok.head == root and "aux" in tok.dep_)]

    full_pred = sorted([*aux, root], key=lambda tok: tok.i)
    has_passive_aux = any("pass" in pred.dep_ for pred in aux)

    return full_pred, has_passive_aux

    
def linearise_tree(sentence: spacy.tokens.Span) -> T.List[T.Tuple[str, str, int]]:
    sent_start = sentence.start
    return [(tok.text, tok.dep_, tok.head.i - sent_start) for tok in sentence]


def find_dependent(sentence: spacy.tokens.Span, tok_id: int, dep: T.Union[str, T.Set[str]]):
    """Find children of tok with id `tok_id` that have certain dep(s)"""
    if isinstance(dep, str):
        dep = {dep}

    head = sentence[tok_id]
    
    return [tok for tok in sentence
            if tok.head == head and tok.dep_ in dep] or None


def save_file(df, result_name: str, path=RESULTS_PATH):
    filename = Path(path) / result_name
    df.to_csv(filename, index=False)


def clean_text(text: str) -> str:
    return text.replace("\xa0", " ").strip()


def find_sentences(texts):
    nlp, matcher = get_models()

    sentences = []
    for text in tqdm(texts):
        # text = text.replace("\n", " ").strip()
        text = clean_text(text)
        doc = nlp(text)
        for i, sentence in enumerate(doc.sents):
            matches = matcher(sentence)
            if matches:
                # int ids of the elements
                match = matches[0]
                subject = match[1][1]
                root = match[1][0]
                distractor  = match[1][2]

                root_tok, subject_tok, distractor_tok = (
                        sentence[root], sentence[subject], sentence[distractor])
                root_number = root_tok.morph.get("Number")
                if not root_number:     # skips non-agreeing roots
                    continue

                if subject < distractor and distractor < root and root != "нет":
                    info = [(word.text, word.lemma_, word.pos_, word.dep_) for word in sentence]
                    tokens, lemmas, pos, deps = list(zip(*info))

                    short_version_toks = shorten_sentence(sentence, subject, distractor, root)
                    short_version = " ".join([tok.text for tok in short_version_toks])
                    short_version_ids = [tok.i - sentence.start for tok in short_version_toks]

                    # short version that retains (some) clausal modifiers
                    short_version_toks_2 = shorten_sentence(sentence, subject, distractor, root, keep_clausal=True)
                    short_version_2 = " ".join([tok.text for tok in short_version_toks_2])
                    short_version_ids_2 = [tok.i - sentence.start for tok in short_version_toks_2]

                    predicate_toks, has_passive_aux = get_full_predicate(sentence, subject, distractor, root)

                    sentences.append({
                        "i": i,
                        "sentence": sentence,
                        "tree": linearise_tree(sentence),
                        "subject": subject_tok,
                        "subject_id": subject,
                        "subject_number": subject_tok.morph.get("Number"),
                        "subject_pos": subject_tok.pos_,
                        "subject_has_amod": find_dependent(sentence, subject, "amod"),
                        "distractor": distractor_tok,
                        "distractor_id": distractor,
                        "distractor_number": distractor_tok.morph.get("Number"),
                        "distractor_pos": distractor_tok.pos_,
                        "distractor_has_amod": find_dependent(sentence, distractor, "amod"),
                        "root": root_tok,
                        "root_id": root,
                        "root_number": root_number,
                        "root_pos": root_tok.pos_,
                        "lemmas": lemmas,
                        "tokens": tokens,
                        "pos": pos,
                        "short_version": short_version,
                        "short_version_ids": short_version_ids,
                        "short_version2": short_version_2,
                        "short_version2_ids": short_version_ids_2,
                        "predicate": [tok.text for tok in predicate_toks],
                        "has_passive_aux": has_passive_aux,
                    })

    df = pd.DataFrame(sentences)
    return df

if __name__ == "__main__":
    # dataset = download_dataset()
    # dataset.save_to_disk(str(PROJECT_PATH))
    dataset2 = datasets.Dataset.load_from_disk(str(PROJECT_PATH))

    shuffle=True
    skip=0
    take=60000
    texts = iterate_dataset(dataset2, shuffle=shuffle, skip=skip, take=take)
    
    results = find_sentences(texts)
    save_file(results, f"test-2.csv")