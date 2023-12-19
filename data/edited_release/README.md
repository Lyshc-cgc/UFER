# Edited Release

The 'edited_release' folder contains the data edited from [**the Original data**](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html). 

1. We manually checked and replaced the uninterrupted spaces (i.e., \u00A0) within the original data with hyphens. 
2. We add more detailed explanations for each file.
3. We manually deleted some incomprehensible instances.

>Original Paper: [Choi E, Levy O, Choi Y, et al. Ultra-Fine Entity Typing[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 87-96.](https://arxiv.org/abs/1807.04905)  

- ./crowd
contains crowdsourced examples (shown in Section 2 of the original paper).

- ./distant_supervision
contains distantly supervised training dataset, from entity linking (el_train.json, el_dev.json) and headwords 
(headword_train.json, headword_dev.json).
  - Shown in Section 3.1 of the original paper, entity linking dataset obtains entity mentions that were linked to Wikipedia
  in HTML (i.e., exploiting existing hyperlinks in web pages), and extract relevant types from their encyclopedic definitions
  (Section 3.1). In 'el_dev.json' file, some instances have type labels with ontology, so they have non-empty 'goal_y_str'
  and 'goal_y' fields. While others (from line 11070 to end) have type labels without ontology, so they have empty 'goal_y_str'
  and 'goal_y' fields.
  - Shown in Section 3.2 of the original paper, head words dataset extracts nominal head words as type labels with a dependency 
  parser (Manning et al., 2014) from the Gigaword corpus as well as the Wikilink dataset.

- ./ontonotes
contains original ontonotes train/dev/test dataset from https://github.com/shimaokasonse/NFGEC, as well as newly augmented
training dataset.  As for the augmented training dataset, it's augmented with Wikipedia definition sentences (WIKI) and
head word supervision (HEAD), which has more instances and more coverage of labels. 

- ./ontology
contains other files, notably the type ontology (89 types) used in the ontonotes dataset. 'char_vocab.english.txt' is 
the character  'type.txt' file consists of:
  - 9 general type (line 1 to line 9): person, location, object, organization, place, entity, object, time, event
  - 121 fine-grained types (line 10 to line 130): mapped to fine-grained entity labels from prior work (Ling and Weld,2012;
  Gillick et al., 2014) (e.g. film, athlete)
  - 10201 ultra-fine types (line 131 to line10331), including every other label in the type space (e.g. detective, lawsuit,
  temple, weapon, composer)

There are 3 granularities (general, fine, ultra-fine) in the crowd, headword and entity linking dataset, while there are
only general and fine-grained type in ontonotes dataset.