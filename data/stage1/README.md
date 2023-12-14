# Stage 1

Stage 1 is the first stage of the processor. It takes the original data (in the `data/edited_release` folder) and 
parse all spans (NP span and named entity span) using [Stanza](https://stanfordnlp.github.io/stanza/) and 
[spaCy](https://spacy.io/). The processed data is in the `data/stage1` folder.