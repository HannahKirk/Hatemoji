
# Hatemoji

This repo contains the two datasets from our 2021 paper: _Hatemoji: A Test Suite and Adversarially-Generated Dataset for Benchmarking and Detecting Emoji-based Hate_. https://arxiv.org/abs/2108.05921


## Citation
If you use either of these datasets please cite as 'Kirk, H. R., Vidgen, B., Rottger, P., Thrush, T., & Hale, S. A. (2021). Hatemoji: A Test Suite and Adversarially-Generated Dataset for Benchmarking and Detecting Emoji-based Hate.'

Contact Hannah if you have feedback or queries: hannah.kirk@oii.ox.ac.uk.


## HatemojiCheck

* HatemojiCheck is a test suite of 3,930 test cases covering seven functionalities of emoji-based hate and six identities. 
* `HatemojiCheck.csv` contains the text for each test case and its gold-standard label from majority agreement of three annotators. We provide labels by target of hate. 
* HatemojiCheck can be used to evaluate the robustness of hate speech classifiers to constructions of emoji-based hate. 

## HatemojiTrain

* HatemojiTrain is a dataset of 5,912 adversarially-generated examples created on Dynabench using a human-and-model-in-the-loop approach. We collect data in three consecutive rounds. 
* Our work follows on from Vidgen et al (2021) _Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection_ (http://arxiv.org/abs/2012.15761) who collect four rounds of textual adversarial examples. The R1-R4 data is avaliable at https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset.
* The entries in `HatemojiTrain.csv` are labeled by round (R5-7). The text of each entry is given with its gold-standard label from majority agreement of three annotators. Each original entry is associated with a perturbation so each row of the `.csv` matches these two cases. We also provide granular labels of type and target for hateful entries. 
* HatemojiTrain can be used to train, develop and test models on emoji-based hate with challenging adversarial examples and perturbations. 

### Columns in `HatemojiCheck.csv`:

**case_id**: The unique ID of the test case (assigned to each of the 3,930 cases generated)

**templ_id**: The unique ID of the template (original=.0, identity perturbation=.1, polarity perturbation=.2, emoji perturbation = .3) from which the test case was generated 

**test_grp_id**: The ID of the set of templates (original, identity perturbation, polarity perturbation, no emoji perturbation) from which the test case was generated.

**text**: The text of the test case.

**target**: Where applicable, the protected group targeted or referenced by the test case. We cover six protected groups in the test suite: women, trans people, gay people, black people, disabled people and Muslims.

**functionality**: The shorthand for the functionality tested by the test case.

**set**: Whether the test case is an original statement, a identity perturbation, a polarity perturbation or a no emoji perturbation.

**label_gold**: The gold standard label (hateful/non-hateful) of the test case. All test cases within a given functionality have the same gold standard label.

**unrealistic_flags**: The number of annotators (/3) who flagged the test case as unrealistic.

**included_in_test_suite**: Indicator for whether test case is included in final HatemojiCheck test suite. All 3,930 test cases are included. 


### Columns in `HatemojiTrain.csv`:

**entry_id**: The unique ID of the entry (assigned to each of the 5,912 cases generated).

**text**: The text of the entry.

**type**: The type of hate assigned to hateful entries.

**target**: The target of hate assigned to hateful entries.

**round.base**: The round where the entry was generated.

**round.set**: The round and whether the entry came from an original statement (a) or a perturbation (b).

**set**: Whether the entry is an original or perturbation.

**split**: The randomly-assigned train/dev/test split using in our work (80:10:10).

**label_gold**: The gold standard label (hateful/non-hateful) of the test case.

**matched_text**: The text of the paired perturbation. Each original entry has one perturbation.

**matched_id**: The unique entry ID of the paired perturbation.

## Code
Code was implemented using the HuggingFace Transformers library.
