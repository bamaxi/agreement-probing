from ast import literal_eval
import typing as T
from pathlib import Path

import pandas as pd
import pymorphy3
from pymorphy3 import MorphAnalyzer
from tqdm.auto import tqdm


morph_parser = MorphAnalyzer()


OPTIONS_4 = [
    ("sing", "sing"),
    ("sing", "plur"),
    ("plur", "sing"),
    ("plur", "plur")
]
DEPS_TO_INFLECT = {"amod"}

ID_COL = "id"
SENTENCE_COL = "sentence"
CHANGED_SENTENCE_COL = "changed_sentence"
TOKENS_COL = "tokens"
TREE_COL = "tree"
POS_COL = "pos"

SHORT_SENTENCE_COL = "short_version"
SHORT_SENTENCE_IDS_COL = "short_version_ids"

SHORT_SENTENCE_COL_2 = "short_version2"
SHORT_SENTENCE_IDS_COL_2 = "short_version2_ids"
SHORT_2 = [SHORT_SENTENCE_COL_2, SHORT_SENTENCE_IDS_COL_2]

ROOT_FORM_COL = "root"
ROOT_ID_COL = "root_id"
ROOT_NUMBER_COL = "root_number"
ROOT_POS_COL = "root_pos"

SUBJECT_FORM_COL = "subject"
SUBJECT_ID_COL = "subject_id"
DISTRACTOR_FORM_COL = "distractor"
DISTRACTOR_ID_COL = "distractor_id"
SUBJECT_NUMBER_COL = "subject_number"
DISTRACTOR_NUMBER_COL = "distractor_number"
SUBJECT_POS_COL = "subject_pos"
DISTRACTOR_POS_COL = "distractor_pos"

# SUBJECT_HOMONYMS_COL = "subject_homonyms"
COMMON_DISTRACTOR_HOMONYMS_COL = "distractor_homonyms"
# TARGET_SUBJECT_HOMONYMS_COL = "target_subject_homonyms"
AGR_DISTRACTOR_HOMONYMS_COL = "agr_distractor_homonyms"
AGREE_EQUALLY_SUBJECT_DISTRACTOR = "agree_equally"
TUPLE_4_HAS_HOMONYMS = "tuple_4_has_homonyms"

INFLECTED_MODIFIERS_COL = "inflected_modifiers"
MODIFIERS_INFLECTION_ERRORS = "modifiers_inflection_errors"
SUBJECT_INFLECTED_MODIFIERS_COL = "subject_inflected_modifiers"
DISTRACTOR_INFLECTED_MODIFIERS_COL = "distractor_inflected_modifiers"
SUBJECT_MODIFIERS_INFLECTION_ERRORS = "subject_modifiers_inflection_errors"
DISTRACTOR_MODIFIERS_INFLECTION_ERRORS = "distractor_modifiers_inflection_errors"

EVAL_COLS = []

KEYS = [
    SENTENCE_COL, TOKENS_COL, TREE_COL, POS_COL,
    SHORT_SENTENCE_COL, SHORT_SENTENCE_IDS_COL,
    ROOT_FORM_COL, ROOT_ID_COL, ROOT_NUMBER_COL, ROOT_POS_COL,
    SUBJECT_FORM_COL, DISTRACTOR_FORM_COL,
    SUBJECT_ID_COL, DISTRACTOR_ID_COL,
    SUBJECT_NUMBER_COL, DISTRACTOR_NUMBER_COL,
    SUBJECT_POS_COL, DISTRACTOR_POS_COL
]

LITERAL_EVAL_COLS = [
    TREE_COL, POS_COL,
    ROOT_NUMBER_COL, SUBJECT_NUMBER_COL, DISTRACTOR_NUMBER_COL,
    SHORT_SENTENCE_IDS_COL,
]
TUPLE_TO_LIST_COLS = [
    TOKENS_COL,
]

KIND_COL = "kind"

FILENAME_APPENDIX = "_perturbed2"


PARTICIPLE_POS = {"PRTF", "PRTS"}
PYMORPHY_VERBAL_POS = {"VERB"} | PARTICIPLE_POS
PYMORPHY_ADJ_POS = {"ADJF", "ADJS"}
PYMORPHY_NOMINAL_POS = {"NOUN", "PRON"}
CONLLU_UPOS2PYMORPHY_POS = {
    "VERB": PYMORPHY_VERBAL_POS | PYMORPHY_ADJ_POS,
    "AUX": PYMORPHY_VERBAL_POS | PYMORPHY_ADJ_POS,
    "ADJ": {"NUMR"} | PYMORPHY_ADJ_POS | PARTICIPLE_POS,
    "NUM": {"NUMR"} | PYMORPHY_ADJ_POS,
    "DET": PYMORPHY_ADJ_POS,
    "PRON": {"NPRO"} | PYMORPHY_ADJ_POS,
    "NOUN": {"NOUN", "NPRO"},
    "PROPN": {"NOUN"},
}

CONLLU_FEAT2PYMORPHY_TAGFEAT = {
    "nom": "nomn", "gen": "gent", "par": "gen2", "dat": "datv", "acc": "accs",
        "ins": "ablt", "loc": "loct", "voc": "voct",
    "fem": "femn",
    "1": "1per", "2": "2per", "3": "3per",
    "fut": "futr",
}
PYMORPHY_TAGFEAT2CONLLU_FEAT = {val: key for key, val in CONLLU_FEAT2PYMORPHY_TAGFEAT.items()}


LinearisedTree = T.List[T.Tuple[str, str, int]]


def tokenize(sentence: str):
    return sentence.split()


def extract_span(
    sentence: str, tokens: T.Tuple[str], subject_id: int, distractor_id: int
) -> T.List[str]:
    tokens = tokenize(sentence)

    return tokens[:distractor_id + 3]


def first_letter_to_upper(strings: T.List[str]):
    first_item = strings[0]
    if strings[0].islower():
        strings[0] = first_item[0].upper() + first_item[1:]


def change_subject_distr(
    sentence: str, tokens: T.Tuple[str], 
    subject_id: int, distractor_id: int,
    new_subject: str, new_distractor: str,
    try_change_amod: bool=True, try_sentence_case: bool=True
) -> T.List[str]:
    # tokens = extract_span(sentence, tokens, subject_id, distractor_id)
    
    tokens[subject_id] = new_subject
    tokens[distractor_id] = new_distractor 

    if try_sentence_case:
        first_letter_to_upper(tokens)

    return tokens


def linearise_sent(elements: T.Iterable[str]):
    return " ".join(elements)


def unify_alphabet(
    sentence: str, translation: T.Mapping[int, int] = str.maketrans(dict(zip("Ёё", "Ее")))
) -> str:
    sentence = sentence.translate(translation)
    return sentence


HomonymsGrammemes = T.List[T.Set[str]]
def find_paradigm_homonyms(parse: pymorphy3.analyzer.Parse) -> HomonymsGrammemes:
    """Find if the form has homonyms in other cells of the lexeme paradigm

    "Other cells the of lexeme paradigm" means other inflection features"""
    orig_lexeme = parse.normal_form
    orig_word = parse.word
    orig_tag = parse.tag

    paradigm_homonyms = []
    # this now checks forms disregarding `е` / `ё` distinction
    for _p in morph_parser.parse(unify_alphabet(parse.word)):
        if (_p.normal_form == orig_lexeme
                and unify_alphabet(_p.word) == unify_alphabet(orig_word)
                and _p.tag != orig_tag):
            paradigm_homonyms.append(set(_p.tag.grammemes))

    return paradigm_homonyms


def filter_AccPl_homonyms(homonyms: HomonymsGrammemes):
    return {hom for hom in homonyms if hom.issuperset({"acc", "pl"})}


def choose_agr_feats(
    # real_controller: pymorphy3.analyzer.Parse, 
    agreer: pymorphy3.analyzer.Parse
):
    
    return (agreer and "sing" in agreer.tag and {str(agreer.tag.gender), }) or set()


def find_homonyms_potential_agree(
    feature_value: str, homonyms: HomonymsGrammemes, other_necessary_feats: T.Set[str] =set(),
) -> HomonymsGrammemes:
    """Find whether after a change of `morph_feature` in agreer to have value `feature_value`,
    it could have potentially agreed not with real controller but with homonymous form

    Example:
        `пробки не потребуется` (orig)  
        `пробки не потребуются` (agreer number changed to `plur`)  
        here controller `пробки` (gen.sg) is homonymous with (nom.pl)
        and `plur` is exactly the form of the changed agreer
    """

    potential_agreeing_homonyms = []
    for homonym in homonyms:
        if feature_value in homonym and other_necessary_feats.issubset(homonym):
            potential_agreeing_homonyms.append(homonym)

    return potential_agreeing_homonyms


def agree_equally(subject: pymorphy3.analyzer.Parse, distractor: pymorphy3.analyzer.Parse) -> bool:
    return (subject.tag.number == distractor.tag.number
            and (
                subject.tag.number == "plur"
                or subject.tag.gender == distractor.tag.gender
            )
    )


def get_suitable_parse(
    parses: T.List[pymorphy3.analyzer.Parse],
    reference: T.Dict[str, T.Dict[str, int]] = None, ref_feats: T.Dict[str, str] = None,
    ref_upos: str=None,
    is_inflectable_: T.Callable[[pymorphy3.analyzer.Parse], bool] = None,
    # additional_checks: List[Callable[[pymorphy3.analyzer.Parse, Dict[str, str]], bool]] = None
) -> T.Optional[pymorphy3.analyzer.Parse]:
    """Finds among pymophy parses the most similar to `reference` and likely to inflect

    Although taking first pymorphy parse is often enough, there are cases where
    such heuristic helps, especially so in agreement.
    Suppose, we want to change gender of the adjective `ярославский` in the
    sentence `Он поступил в Ярославский университет`. First parse is nominative,
    and the second is accusative. Heuristics defined here correctly find second parse.
    """
    if is_inflectable_ is None:
        is_inflectable_ = lambda p: True

    # animacy is needed to inflect but not necessary here as we have
    #   token + gender + case, this is enough

    NO_CHECK_FEATS = ("verbform", "variant", "animacy")
    if not ref_feats and not reference:
        return None
    if not ref_feats:
        ref_feats = {feat: val for feat, val in reference.items() if val}
    ref_feats = {feat: val for feat, val in ref_feats.items()
                 if feat not in NO_CHECK_FEATS}
    if not ref_feats:
        return None

    ref_pos = CONLLU_UPOS2PYMORPHY_POS.get(ref_upos.upper())
    if not ref_pos:
        # print(f"no pos: {ref_upos}")
        return None

    # maps indices of pymorphy parses to a map describing them:
    #   {feat: whether the parse inflects for it and equals reference}
    parse_i2feats_eval: T.Dict[int, T.Dict[str, int]] = {}
    for i, parse in enumerate(parses):
        for feat, ref_val in ref_feats.items():
            parse_val = getattr(parse.tag, feat) or False
            does_val_exist_and_eq = (
                parse_val
                and parse_val == CONLLU_FEAT2PYMORPHY_TAGFEAT.get(ref_val, ref_val)
            )
            parse_i2feats_eval.setdefault(i, {})[feat] = does_val_exist_and_eq

        parse_i2feats_eval[i]["is_inflectable"] = is_inflectable_(parse)

    # print(parse_i2feats_eval)

    # parses with highest number of existing and same-valued feats that may inflect
    sorted_by_criteria_sum = sorted(
        parse_i2feats_eval,
        key=lambda k: sum(parse_i2feats_eval[k].values()), reverse=True
    )

    best_parse = parses[sorted_by_criteria_sum[0]]
    best_parses = [i for i in sorted_by_criteria_sum
                    if parses[i].tag.POS in ref_pos]
    if best_parses:
        best_parse = parses[best_parses[0]]
    else:
        # print(f"no proper pos ({ref_upos}, {ref_pos}): {parses}")
        return None

    if not parse_i2feats_eval[sorted_by_criteria_sum[0]]["is_inflectable"]:
        return None

    # print(f"best parse: `{best_parse}`")
    return best_parse


def inflect_modifiers(
    sentence: str, tokens: T.Tuple[str], tree: LinearisedTree, pos: T.Tuple[str],
    subject_id: int, distractor_id: int,
    orig_subject_number: str, orig_distractor_number: str,
    target_subject_number: T.Optional[str]=None, target_distractor_number: T.Optional[str]=None
):
    general_inflected = {}
    general_inflection_errors = {}

    # print(sentence, tokens, tree, pos, sep="\n")

    heads = zip(
        ["subject", "distractor"],
        [subject_id, distractor_id],
        [orig_subject_number, orig_distractor_number],
        [target_subject_number, target_distractor_number]
    )

    # print(heads)
    for (kind, head_tok_id, orig_number, target_number) in heads:
        if not target_number:
            continue
        
        inflected = general_inflected.setdefault(kind, [])
        inflection_errors = general_inflection_errors.setdefault(kind, [])

        # print(tree)

        for tok_id, (tok_form, tok_dep, tok_head) in enumerate(tree):
            tok_pos = pos[tok_id]

            if tok_head == head_tok_id and tok_dep in DEPS_TO_INFLECT:
                parse = get_suitable_parse(
                    morph_parser.parse(tok_form), ref_feats={"number": orig_number},
                    ref_upos=tok_pos
                )
                # print(tok_form, tok_dep, tok_head, target_number)
                # assert parse    # to err or not to err?
                if not parse:
                    inflection_errors.append((tok_id, tok_form, f"no form: {target_number}"))
                    continue

                new_tok = parse.inflect({target_number})
                if not new_tok:
                    inflection_errors.append((tok_id, tok_form, f"no form: {target_number}"))
                    continue

                tokens[tok_id] = new_tok.word
                # inflected.append(new_tok.word)
                inflected.append(new_tok.word)
    
    return general_inflected, general_inflection_errors


def sort_set(s: T.Set[T.Any]):
    return tuple(sorted(s))


def make_quadruplets(
    i: int,
    sentence: str, tokens: T.Tuple[str], tree: LinearisedTree, pos: T.Tuple[str],
    short_version: str, short_version_ids: T.List[int],
    root: str, root_id: int, root_number: T.List[str], root_pos: str,
    subject: str, distractor: str, subject_id: int, distractor_id: int,
    subject_number: T.List[str], distractor_number: T.List[str],
    subject_pos: str, distractor_pos: str,
    short_version2: T.Optional[str]=None, short_version2_ids: T.Optional[T.List[int]] = None,
    sent_info: T.Optional[T.Dict[str, T.Any]]=None,
    **kwargs
) -> T.List[T.Dict[str, T.Union[str, int, T.List[str], T.List[int], LinearisedTree]]]:
    if not (subject_number and distractor_number and root_number):
        return []
    subject_number = subject_number[0].lower()
    distractor_number = distractor_number[0].lower()
    root_number = root_number[0].lower()

    # we trust the original markup and select the pymorphy analysis which corresponds to it
    orig = {
        ID_COL: i,
        SENTENCE_COL: sentence, TOKENS_COL: tokens, TREE_COL: tree,
        SHORT_SENTENCE_COL: short_version, SHORT_SENTENCE_IDS_COL: short_version_ids,
        SUBJECT_FORM_COL: subject, DISTRACTOR_FORM_COL: distractor,
        SUBJECT_NUMBER_COL: subject_number, DISTRACTOR_NUMBER_COL: distractor_number,
        # CHANGED_SENTENCE_COL: " ".join(extract_span(sentence, subject_id, distractor_id)),
        CHANGED_SENTENCE_COL: " ".join(tokens),
        KIND_COL: "orig",
        "subject_changed": False, "distractor_changed": False,
    }

    # print(sentence)
    # print(subject, subject_number, distractor, distractor_number)

    subject_analyses = morph_parser.parse(subject)
    distractor_analyses = morph_parser.parse(distractor)
    root_analyses = morph_parser.parse(root)

    subject_correct_parse = get_suitable_parse(
        subject_analyses, ref_feats={"number": subject_number}, ref_upos=subject_pos)
    distractor_correct_parse = get_suitable_parse(
        distractor_analyses, ref_feats={"number": distractor_number}, ref_upos=distractor_pos)
    root_correct_parse = get_suitable_parse(
        root_analyses, ref_feats={"number": root_number}, ref_upos=root_pos)

    # print(subject_correct_parse, distractor_correct_parse)

    if not (subject_correct_parse and distractor_correct_parse and root_correct_parse):
        return []
    
    # subject_homonyms = find_paradigm_homonyms(subject_correct_parse)
    distractor_homonyms = find_paradigm_homonyms(distractor_correct_parse)
    target_distractor_homonyms = find_homonyms_potential_agree(
        root_number, distractor_homonyms, choose_agr_feats(root_correct_parse)
    )

    orig.update({
        # SUBJECT_HOMONYMS_COL: subject_homonyms,
        COMMON_DISTRACTOR_HOMONYMS_COL: sort_set(distractor_homonyms) or None,
        AGR_DISTRACTOR_HOMONYMS_COL: sort_set(target_distractor_homonyms) or None,
        AGREE_EQUALLY_SUBJECT_DISTRACTOR: agree_equally(subject_correct_parse, distractor_correct_parse)
    })

    print(subject_correct_parse, distractor_correct_parse)

    subject_ana_number = subject_correct_parse.tag.number
    distractor_ana_number = distractor_correct_parse.tag.number

    # assert subject_ana_number == subject_number[0].lower()
    # assert distractor_ana_number == distractor_number[0].lower()

    _orig = sent_info.copy()
    # _orig.update(orig)
    results = [{**_orig, **orig}]
    tuple_4_homonyms = []
    for subject_number_to, distractor_number_to in OPTIONS_4:
        if (subject_ana_number != subject_number_to
                or distractor_ana_number != distractor_number_to):
            
            new_subj = subject_correct_parse.inflect({subject_number_to})
            new_distr = distractor_correct_parse.inflect({distractor_number_to})

            print(new_subj, new_distr)

            is_subject_changed = new_subj != subject_correct_parse
            is_distractor_changed = new_distr != distractor_correct_parse

            # if not new_subj or not new_distr:
            #     res = {
            #         SENTENCE_COL: sentence, TOKENS_COL: tokens, TREE_COL: tree,
            #         SUBJECT_FORM_COL: None, DISTRACTOR_FORM_COL: None,
            #         SUBJECT_NUMBER_COL: subject_ana_number, DISTRACTOR_NUMBER_COL: distractor_ana_number,
            #         CHANGED_SENTENCE_COL: None, SHORT_SENTENCE_COL: None,
            #         KIND_COL: "chng"
            #     }
            # else: 
            if new_subj and new_distr:
                this_tokens = tokens.copy()

                args = (
                    subject_number_to if is_subject_changed else None,
                    distractor_number_to if is_distractor_changed else None,
                )
                inflected_modifiers, modifiers_inflection_errors = inflect_modifiers(
                    sentence, this_tokens, tree, pos,
                    subject_id, distractor_id,
                    subject_ana_number, distractor_ana_number,
                    *args
                )

                changed_sent_part = change_subject_distr(
                    sentence, this_tokens,
                    subject_id, distractor_id,
                    new_subj.word, new_distr.word
                )

                short_version = " ".join([this_tokens[i] for i in short_version_ids])

                res = {
                    ID_COL: i,
                    SENTENCE_COL: sentence, TOKENS_COL: this_tokens, TREE_COL: tree,
                    SHORT_SENTENCE_COL: short_version, SHORT_SENTENCE_IDS_COL: short_version_ids,
                    SUBJECT_FORM_COL: subject, DISTRACTOR_FORM_COL: distractor,
                    SUBJECT_NUMBER_COL: subject_ana_number, DISTRACTOR_NUMBER_COL: distractor_ana_number,
                    f"target_{SUBJECT_FORM_COL}": new_subj.word, 
                    f"target_{DISTRACTOR_FORM_COL}": new_distr.word,
                    f"target_{SUBJECT_NUMBER_COL}": new_subj.tag.number, 
                    f"target_{DISTRACTOR_NUMBER_COL}": new_distr.tag.number,
                    "subject_changed": new_subj.word != subject,
                    "distractor_changed": new_distr.word != distractor,
                    CHANGED_SENTENCE_COL: " ".join(changed_sent_part),
                    # INFLECTED_MODIFIERS_COL: inflected_modifiers or None,
                    # MODIFIERS_INFLECTION_ERRORS: modifiers_inflection_errors or None,
                    SUBJECT_INFLECTED_MODIFIERS_COL: inflected_modifiers.get("subject") or None,
                    SUBJECT_MODIFIERS_INFLECTION_ERRORS: modifiers_inflection_errors.get("subject") or None,
                    DISTRACTOR_INFLECTED_MODIFIERS_COL: inflected_modifiers.get("distractor") or None,
                    DISTRACTOR_MODIFIERS_INFLECTION_ERRORS: modifiers_inflection_errors.get("distractor") or None,
                    KIND_COL: "chng"
                }

                new_distractor_homonyms = find_paradigm_homonyms(new_distr)
                # TODO: change to new_root when inflecting root
                target_distractor_homonyms = find_homonyms_potential_agree(
                    root_number, new_distractor_homonyms, choose_agr_feats(root_correct_parse)
                )
                sorted_target_distractor_homonyms = sort_set(target_distractor_homonyms)
                res.update({
                    COMMON_DISTRACTOR_HOMONYMS_COL: sort_set(distractor_homonyms) or None,
                    AGR_DISTRACTOR_HOMONYMS_COL: sorted_target_distractor_homonyms or None,
                    AGREE_EQUALLY_SUBJECT_DISTRACTOR: agree_equally(new_subj, new_distr)
                })
                if target_distractor_homonyms:
                    tuple_4_homonyms.append(sorted_target_distractor_homonyms)

                if short_version2 and short_version2_ids:
                    # print(short_version2_ids)
                    short_version2 = " ".join([this_tokens[i] for i in short_version2_ids])
                    short_res_2 = {SHORT_SENTENCE_COL_2: short_version2, SHORT_SENTENCE_IDS_COL_2: short_version2_ids}

                    res.update(short_res_2)

                _res = sent_info.copy()
                # _res.update(res)
                results.append({**_res, **res})

    # print(*results, sep="\n\t")
    if len(results) != 4:
        return []
    
    if tuple_4_homonyms:
        for res in results:
            res[TUPLE_4_HAS_HOMONYMS] = tuple_4_homonyms

    return results


def parse_dicts(
    dicts: T.Iterable[T.Dict[str, str]],
    n_max: int=float("inf")
):
    results = []

    for i, dict_ in enumerate(dicts):
        # print(dict_)
        if i >= n_max:
            break
        
        result = make_quadruplets(i, **dict_, sent_info=dict_)
        # print(result)
        results.extend(result)

    return results


def tuple_str_to_list(x: str):
    return list(literal_eval(x))


def read_from_csv(
    filename: T.Union[str, Path],
    subject_form_col: str = SUBJECT_FORM_COL,
    distractor_form_col: str = DISTRACTOR_FORM_COL,
    subject_number_col: str = SUBJECT_NUMBER_COL,
    distractor_number_col: str = DISTRACTOR_NUMBER_COL
):
    table = pd.read_csv(filename).rename(columns=dict(
        subject_form_col = subject_form_col,
        distractor_form_col = distractor_form_col,
        subject_number_col = subject_number_col,
        distractor_number_col = distractor_number_col
    ))

    for col in TUPLE_TO_LIST_COLS:
        table[col] = table[col].apply(tuple_str_to_list)
    for col in LITERAL_EVAL_COLS:
        table[col] = table[col].apply(literal_eval)

    return table


def parse_csv_data(
    filename: T.Union[str, Path], max_n: T.Optional[int]=None,
    # TODO: change in prod
):
    table = read_from_csv(filename)
    
    if max_n is not None:
        table = table[:max_n]
    print(table.columns)
    print(KEYS)
    
    table_small = table[KEYS]

    dicts = table_small.to_dict("records")
    # print(dicts[0])

    # results = parse_dicts(tqdm(dicts), args.N)
    results = parse_dicts(tqdm(dicts))
    print(len(table), len(results))
    # final_table = pd.concat([table_small, pd.DataFrame(results, index=table.index)], axis=1, join="inner")
    
    results_df = pd.DataFrame(results)
    # print(results_df.columns)

    # ON_COL = SENTENCE_COL
    # cols_to_replace = [col for col in results_df.columns if col in KEYS]
    # first_merge_cols = [col for col in results_df.columns
    #                     if (col not in KEYS) or col == ON_COL]
    # # final_table = table.combine_first(results_df)
    # # print(cols_to_replace)
    # # print(first_merge_cols)

    # final_table = pd.merge(
    #     table,
    #     results_df[first_merge_cols],
    #     how="inner", on=ON_COL
    # )
    # final_table[cols_to_replace] = results_df[cols_to_replace]
    # print(final_table[["sentence", "tokens", "kind", 'subject', 'distractor']][:2].to_dict())
    first_cols = [
        'id', 'sentence', 'tokens', 'tree', 'pos',
        'root', 'root_id', 'root_number', 'root_pos',
        'subject', 'distractor', 'subject_id', 'distractor_id',
        'subject_number', 'distractor_number', 'subject_pos', 'distractor_pos',
        'changed_sentence', 'kind',
        'subject_changed', 'distractor_changed'
    ]
    final_table = results_df
    final_table = final_table[
        [*first_cols, *[col for col in final_table.columns if col not in first_cols]]
    ]

    print(final_table.columns)

    out_filename = filename.with_stem((str(filename.stem) + FILENAME_APPENDIX))
    final_table.to_csv(out_filename, index=False)

    return table_small


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates quadruplets")

    default_n = None
    parser.add_argument("path", type=str, help="file to parse")
    parser.add_argument("--N", type=int, default=default_n, help="max entries to check")
    # parser.add_argument(
    #     "--short2", action="store_true",
    #     help="apply perturbation to `short_version2` column (`short_version2_ids` must be present)"
    # )
    args = parser.parse_args()

    # if args.short2:
        # print("will perturb short_version2 too (using short_version2_ids)")
    KEYS.extend(SHORT_2)
    LITERAL_EVAL_COLS.append(SHORT_SENTENCE_IDS_COL_2)

    # "./arseny_data/results_wiki5.csv"
    parse_csv_data(Path(args.path), max_n=args.N)
