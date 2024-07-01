# %%
import typing as T
from pathlib import Path
import pandas as pd

# %%
import natasha
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

COL_TEXT_TO_SHORTEN = "short_version"

N_NEW_COLS = 4
DEFAULT_RETURN = pd.Series([0] * N_NEW_COLS)

FILENAME_APPENDIX = "_shortened"

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

def create_doc(text):
    doc = Doc(text)

    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    doc.parse_syntax(syntax_parser)
    
    return doc

# %%
def short_id(id_natasha: str):
    return int(id_natasha[2:])


def to_str(tokens: T.List[natasha.doc.DocToken], join_on: str=" "):
    return join_on.join([tok.text for tok in tokens])


def filter_with_natasha(row, col_text_to_shorten=COL_TEXT_TO_SHORTEN):
    '''
    возвращает pd.Series([short_version2, misagreed_id, predicate])
    
    short_version2 : str : более сокращенная версия предложения
    misagreed_id : int : номер токена, где происходит рассогласование
    predicate : list : (ауксильяр) + основной глагол или прилагательное / причастие 
    '''
    if pd.isna(row[col_text_to_shorten]):
        return DEFAULT_RETURN

    text_to_shorten = row[col_text_to_shorten].strip()

    short_version2 = []
    
    # если слились заголовок и предложение в одну строку
    if '\n' in text_to_shorten:
        return DEFAULT_RETURN
    
    doc = create_doc(text_to_shorten)
    
    subj_token = 0
    distr_token = 0
    root_token = 0
    
    for token in doc.tokens:
        if token.text == row['subject']:
            subj_token = token
            
        elif token.text == row['distractor']:
            distr_token = token
        
        elif token.text == row['root']:
            # если что-то ошибочно определилось как глагол или прилагательное
            if (token.pos != 'VERB') and (token.pos != 'AUX') and (token.pos != 'ADJ'):
                return DEFAULT_RETURN
            else:
                root_token = token
    
    if (subj_token == 0) or (distr_token == 0) or (root_token == 0):
        # наташа не может найти какой-то токен по тексту
        print(text_to_shorten)
        print(row['subject'], row['distractor'], row['root'])
        print(subj_token, distr_token, root_token, sep="\n\t")
        print(f"ban (some 0): {text_to_shorten}")
        return DEFAULT_RETURN
    
    # at this point the row is a good one and we can do our further shortening
    print(text_to_shorten)
    print(f"({subj_token}, {subj_token.id}) ({distr_token}, {distr_token.id}) ({root_token}, {root_token.id})")

    prep_token = 0
    aux_token = 0
    words_after_root = []
    misagreed_id = -1
    
    predicate = []

    for token in doc.tokens: 
        # найти предлог, голова предлога — дистрактор
        if (token.pos == 'ADP') and (token.head_id == distr_token.id):
            prep_token = token

        # найти AUX глагол, если он есть (_был_ принят)
        elif (token.lemma in ('быть', 'стать', 'являться')) and (token.head_id == root_token.id):
            aux_token = token

        elif (short_id(token.id) > short_id(root_token.id)):
            words_after_root.append(token)


    subj_capital = subj_token.text[0].upper() + subj_token.text[1:]
    short_version2 = [subj_token]
    short_version2_str_list = [subj_capital]

    if prep_token != 0:
        short_version2.append(prep_token)

    short_version2.append(distr_token)

    if aux_token != 0:
        short_version2.append(aux_token)

    short_version2.append(root_token)

    print(f"before adding `words_after_root`:"
          f"\nshort_version {to_str(short_version2)}"
          f"\nwords_after_root {to_str(words_after_root)}")

    short_version2.extend(words_after_root)
    
    if aux_token != 0:
        misagreed_id = short_version2.index(aux_token)
        predicate.extend([aux_token, root_token])
        
    else:
        misagreed_id = short_version2.index(root_token)
        predicate.append(root_token)

    short_version2_ids = [short_id(tok.id) - 1 for tok in short_version2]
    short_version2_str_list.extend(tok.text for tok in short_version2[1:])

    predicate = [tok.text for tok in predicate]

    return pd.Series([
        ' '.join(short_version2_str_list), short_version2_ids,
        misagreed_id, predicate
    ])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Shortens sentences")

    default_n = None
    parser.add_argument("path", type=str, help="file to parse")
    parser.add_argument("--column", type=str, help="column to use as text", default=COL_TEXT_TO_SHORTEN)
    parser.add_argument("--N", type=int, default=default_n, help="max entries to check")
    args = parser.parse_args()

    # "./arseny_data/results_wiki5.csv"

    filename = Path(args.path)
    df = pd.read_csv(filename)
    print(df.columns)

    if args.N is not None:
        df = df[:args.N]

    df[
        ['short_version2', 'short_version2_ids', 'misagreed_id', 'predicate']
    ] = df.apply(filter_with_natasha, col_text_to_shorten=args.column, axis=1)

    # df = df[df.short_version2 != 0]
    # df = df.drop_duplicates(subset=['short_version2'])

    out_filename = filename.with_stem((str(filename.stem) + FILENAME_APPENDIX))

    df.to_csv(out_filename, index=False)

if __name__ == "__main__":
    main()