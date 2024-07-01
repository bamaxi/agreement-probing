# Data augmentation

The code in this folder allows extracting relevant sentences with agreeement 

- `./wiki_extraction/agreement_data.py` — extracts relevant sentence from wikipedia and annotates them for morphology and syntax
    (uses the [`spacy` package](https://spacy.io/))
- `./perturbation.py` — alternates the number in sentences created by the previous script
    (uses the [`pymorphy3` package](https://github.com/no-plagiarism/pymorphy3))

## Wikipedia data extraction

`agreement_data.py` creates csv file with the following fields

`spacy` follows the Universal Dependencies tagset

### <a name="extraction-fields"></a> Fields

A row describes a sentence from the dataset.

- `i` — index of the sentence among a collection of documents (in the original order of `olm/wikipedia` dataset, shuffled at iteration with order defined by seed `RANDOM_SEED`)
- `sentence` — original sentence, as sentencized by the `spacy` pipeline
- `tree` — linearised sentence tree. A list of 3-tuples `(token form, token relation to its head, id of the token head)`
- `subject` — form of the subject
- `subject_id` — index of the subject within the sentence
- `subject_number` — number of the subject
- `subject_pos` — part of speech of the subject
- `subject_has_amod` — children of the subject with `amod` dependency relation (if any, else empty string). We attempt to change their number too in `perturbation.py`.
- `distractor` — form of the distractor
- `distractor_id` — index of the distractor within the sentence
- `distractor_number` — number of the distractor
- `distractor_pos` — part of speech of the distractor
- `distractor_has_amod` — children of the distractor with `amod` dependency relation (if any, else empty string). We attempt to change their number too in `perturbation.py`.
- `root` — form of the root
- `root_id` — index of the distractor within the sentence
- `root_number` — number of the root
- `root_pos` — part of speech of the root
- `lemmas` — tuple of lemmas of each token in the sentence
- `tokens` — tuple of forms (without whitespace) of each token in the sentence
- `pos` — tuple of parts of speech of each token in the sentence
- `short_version` — short version of the sentence. We generally attempt to keep only necessary dependents, but this version may lack some of those.
- `short_version_ids` — indices of the tokens of the short version. Required by `perturbation.py` to assemble the sentence from changed tokens.
- `short_version2` — alternate short version of the sentence. This version may keep more of the necessary dependents, due to admitting phrases headed by tokens with dependency relation like `*cl*` (this is, minimally, `acl`, `acl:relcl`, sometimes `advcl`).
- `short_version2_ids` — indices of the tokens of the alternate short version. Required by `perturbation.py` to assemble the sentence from changed tokens.
- `predicate` — full predicate of the relevant sentence part. Includes any tokens with `aux*` dependency relation, that are children of root. 
- `has_passive_aux` — True if any of the tokens in `predicate` have `aux:pass` relation.


## Data perturbation

Below we describe only fields that differ somehow from [fields in the extraction file](#extraction-fields).
    Most changes are written to new fields and don't overwrite data in the old fields.

### Fields

A row describes an original sentence from the dataset or an alternated version of the original sentence.

- `i`
- `sentence`
- `tree`
- `subject`
- `subject_id`
- `subject_number`
- `subject_pos`
- `subject_has_amod`
- `distractor`
- `distractor_id`
- `distractor_number`
- `distractor_pos`
- `distractor_has_amod`
- `root`
- `root_id`
- `root_number`
- `root_pos`
- `lemmas`
- `tokens` — tokens of the current version of the sentence (original if `kind`=`orig`, alternated if `kind`=`chng`)
- `pos`
- `short_version` — short version of the alternated sentence
- `short_version_ids`
- `short_version2`  — alternate short version of the alternated sentence
- `short_version2_ids`
- `predicate`
- `has_passive_aux`
- `changed_sentence` — full alternated sentence
- `kind` — kind of the sentence: `orig|chng` (original or changed)
- `distractor_homonyms` — grammatical features of homonyms of the distractor form (as analyzed by `pymorphy3`) (if any, else empty string)
- `agr_distractor_homonyms` — those of the `distractor_homonyms` that could have agreed with root in number and gender. We attempt to check number, and gender too if number is singular. If these exist, then the sentence has a configuration where **attraction** could take place (if `agree_equally` is True and the subject has different number or gender and doesn't agree how a homonym of distractor would agree)
- `agree_equally` — True if subject and distractor in current form has the same agreement features as the subject. These are trivial cases like `Разрешение на строительство было получено давно.`
- `target_subject` — form of the subject after perturbations
- `target_distractor` — form of the distractor after perturbations
- `target_subject_number` — number of the subject after perturbations
- `target_distractor_number` — number of the distractor after perturbations
- `inflected_modifiers` — 
- `modifiers_inflection_errors`
- `subject_inflected_modifiers` — a list of successfully inflected modifiers of the subject.
- `subject_modifiers_inflection_errors` — a list of errors in the inflection of modifiers of the subject. A list of 3-tuples `(id of the modifier, form of the modifier, error message)`
- `distractor_inflected_modifiers` — a list of successfully inflected modifiers of the modifier.
- `distractor_modifiers_inflection_errors`a list of errors in the inflection of modifiers of the distractor. A list of 3-tuples `(id of the modifier, form of the modifier, error message)`
