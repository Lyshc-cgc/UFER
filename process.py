import torch
import spacy
import stanza
import os
import yaml
import multiprocess
import jsonlines
from nltk.tree import Tree
from datasets import load_dataset, concatenate_datasets
from yaml.loader import SafeLoader
from typing import List
from tqdm import tqdm


class Processor:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=SafeLoader)
        self.num_workers = self.config['num_workers']
        self.num_shards = self.config['num_shards']
        self.cuda_device = self.config['cuda_device']
        self.data_dir = self.config['data_dir']

    @staticmethod
    def merge_compound_words(sents: List[List[str]]) -> List[str]:
        """
        1. Some compound words formed by hyphen ('-') had been split into several words, we need to merge them into a
            single word.
            e.g., ['United', '-', 'States'] -> ['United-States'], ['the', 'Semi-', 'finals'] -> ['the Semi-finals']
        2. In stanza, the fraction symbol will be separated by spaces.  We need to merge them together.
            e.g., ['The', 'FMS', '/', 'FMF', 'case'] -> ['The', 'FMS/FMF', 'case'],
                ['3', '/', '4ths', 'to', '9' , '/', '10ths', 'of', 'the', 'tenant', 'farmers', 'on', 'some', 'estates']
                -> ['3/4ths', 'to', '9/10ths', 'of', 'the, 'tenant', 'farmers, 'on', 'some', 'estates']

        :param sents: List[List[str]], a list of sentences, where each sentence is a list of words.
        :return: new_sents, List[str], a list of new sentences
        """
        new_sents = []
        for sent in sents:
            pos = 0  # word position
            while 0 <= pos < len(sent):
                word = sent[pos]
                if word == '-' or (word == '/' and pos >= 1 and sent[pos-1].isdigit()):
                    # e.g., ['a', 'United', '-', 'States', 'b'] -> ['a', 'United-States', 'b']
                    #['3', '/', '4ths'] -> ['3/4ths']
                    if pos - 1 >= 0 and pos + 2 < len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 2])] + sent[pos + 2:]
                    elif pos - 1 >= 0 and pos + 2 >= len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 2])]
                    else:  # pos - 1 < 0, i.e., pos == 0
                        pos += 2  # ignore this word
                    pos = pos - 1  # in this case, the position of the next new word is at the previous position
                elif not word.endswith('B-') and word != '--' and word.endswith('-'):  # e.g., ['a', 'the', 'Semi-', 'finals', 'b'] -> ['a', 'the Semi-finals', 'b']
                    # Special symbols (e.g., '-LRB-', '-LSB-') need to be excluded
                    if pos + 1 == len(sent):  # the last word
                        break
                    elif pos + 2 < len(sent):
                        sent = sent[:pos] + [''.join(sent[pos: pos + 2])] + sent[pos + 2:]
                    else:
                        sent = sent[:pos] + [''.join(sent[pos: pos + 2])]
                    # in this case, the position of the next new word is at the current pos, i.e., pos = pos
                elif not word.endswith('B-') and word != '--' and word.startswith('-'):  # e.g., ['a', 'the', 'Semi', '-finals', 'b'] -> ['a', 'the', 'Semi-finals', 'b']
                    if pos - 1 >= 0 and pos + 1 < len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 1])] + sent[pos + 1:]
                    elif pos -1 >= 0 and pos + 1 >= len(sent):
                        sent = sent[:pos - 1] + [''.join(sent[pos - 1: pos + 1])]
                    else:  # pos - 1 < 0, i.e., pos == 0
                        pos += 2  # ignore this word
                    pos = pos - 1  # in this case, the position of the next new word is at the previous position
                else:
                    pos += 1
            new_sents.append(' '.join(sent))
        return list(set(new_sents))  # remove duplicates

    @staticmethod
    def modify_spacy_tokenizer(nlp):
        """
        Modify the spaCy tokenizer to prevent it from splitting on '-' and '/'.
        Refer to https://spacy.io/usage/linguistic-features#native-tokenizer-additions

        :param nlp: The spaCy model.
        :return: The modified spaCy model.
        """
        from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
        from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
        infixes = (
                LIST_ELLIPSES
                + LIST_ICONS
                + [
                    r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
                    r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                        al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                    ),
                    # Commented out regex that splits on hyphens between letters:
                    # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                    r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                    # r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
                    r"(?<=[{a}])[:<>=/](?=[{a}])".format(a=ALPHA),  # Modified regex to not split words on '/' if it is preceded by a number
                ]
        )
        infix_re = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer
        return nlp

    @staticmethod
    def replace_special_tokens(sent: str) -> str:
        """
        Replace special tokens with original characters.
        e.g., '-LRB-' -> '(', '-RRB-' -> ')', '-LSB-' -> '[', '-RSB-' -> ']'

        :param sent: The sentence to be processed.
        :return: The processed sentences.
        """
        processed_sents = (sent.replace('-LRB-', '(')
                           .replace('-RRB-', ')')
                           .replace('-LSB-', '[')
                           .replace('-RSB-', ']')
                           .replace('-LCB-', '{')
                           .replace('-RCB-', '}')
                           )

        return processed_sents

    def stage1(self, instances, rank = 0, **kwargs):
        """
        Stage1, filter short sentences, get noun phrase (NP) spans and named entity (NE) spans in sentences. English only.
        In this stage, we get flat spans by spaCy. Meanwhile, we get nested spans by spaCy and Stanza.

        :param instances: A batch of instances.
        :param rank: The rank of the current process. It will be automatically assigned a value when multiprocess is
            enabled in the map function.
        :param kwargs:
            1) mode: strict or loose. In strict (default) mode, we get spans based on intersection of spaCy and
                Stanza results. If strict, the span text must be exactly the same as the text in the sentence.
            2) sent_min_len: The number of words in the sentence to be processed must exceed sent_min_len. Default is 10.
            3) span_portion: The proportion of the longest span length to sentence length. To filter out long span.
            4) batch_num_per_device: # Specify number of batches on each device to be processed in map function. Depends on your memory.
            5) batch_size_per_device: Batch size processed in the model on each device.
            6) spacy_model: The spaCy parser config we use.
            7) stanza_model: The stanza parser config we use.
        :return:
        """
        assert kwargs['mode'] in ['strict', 'loose'], f"mode must be one of ('stric', loose)!"
        if self.cuda_device == 'all':
            # set the GPU can be used by stanza in this process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

            # specify the GPU to be used by spaCy, which should be same as above
            spacy.prefer_gpu(rank % torch.cuda.device_count())
        else:
            cuda_devices = str(self.cuda_device).split(',')
            gpu_id = rank % len(cuda_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[gpu_id])

            # specify the GPU to be used by spaCy, which should be same as above
            spacy.prefer_gpu(int(cuda_devices[gpu_id]))

        # load a spaCy and a stanza model in each process
        spacy_nlp = spacy.load(name = kwargs['spacy_model']['name'])
        spacy_nlp = self.modify_spacy_tokenizer(spacy_nlp)  # modify the spaCy tokenizer
        stanza_nlp = stanza.Pipeline(**kwargs['stanza_model'], download_method=None)

        sentences, out_sentences, out_spans = [], [], []

        for idx in range(kwargs['batch_num_per_device']):
            # 1. Some preparations
            # 1.1. get batch sentences for the device in this process
            start_pos, end_pos = idx * kwargs['batch_size_per_device'], (idx + 1) * kwargs['batch_size_per_device']
            raw_sents = instances['sentence'][start_pos: end_pos]

            # 1.2. tokenize the raw sentences using spaCy
            # https://spacy.io/api/language#pipe
            docs = spacy_nlp.pipe(raw_sents, disable = kwargs['spacy_model']['disable'])
            batch_texts = [[token.text for token in doc] for doc in docs]
            batch_texts = list(filter(lambda x: len(x) >= kwargs['sent_min_len'], batch_texts))  # filter short sentences
            if len(batch_texts) <= 0:
                continue

            # 1.3. merge compound words
            # Some compound words formed by hyphen (-) had been split into several words,
            # we need to combine them back into a single word,
            # e.g., ['United', '-', 'States'] -> ['United-States']
            # Also, some fractions or ratio will be separated by spaces, we need to merge them together.
            # e.g., ['3', '/', '4ths', 'to', '9' , '/', '10ths'] -> ['3/4ths', 'to', '9/10ths']
            batch_texts = self.merge_compound_words(batch_texts)

            # 1.4. replace special tokens with original characters
            batch_texts = [self.replace_special_tokens(sent) for sent in batch_texts]

            # 2. process by 2 parsers
            # refer to
            # 1) https://spacy.io/usage/processing-pipelines#processing
            # 2) https://spacy.io/api/language#pipe
            spacy_docs = list(spacy_nlp.pipe(batch_texts))  # covert generator to list

            # Here, spaCy tokenizer will split some words into several tokens according to the rule,
            # even if they are connected together
            # e.g., 'roll-on/roll-off' -> 'roll-on', '/', 'roll-off
            # To ensure consistency of subsequent data, we use the text processed by spaCy as input for stanza
            # batch_texts = [' '.join([token.text for token in s_doc]) for s_doc in spacy_docs]

            # refer to https://stanfordnlp.github.io/stanza/getting_started.html#processing-multiple-documents
            stanza_docs = stanza_nlp.bulk_process(batch_texts)

            for sent, spa_doc, sta_doc in zip(batch_texts, spacy_docs, stanza_docs):
                # 2.1 spaCy
                # 2.1.1 get NP spans by spaCy. They are flat
                # store the start word index, end word index (excluded) and the text of the NP spans.
                # i.e., (start_word_idx, end_word_idx, span_text)
                # The method is included in the spaCy package.
                # refer to https://spacy.io/usage/linguistic-features#noun-chunks
                # and https://spacy.io/api/doc#noun_chunks
                spacy_result = [(chunk.start, chunk.end, chunk.text) for chunk in spa_doc.noun_chunks]

                # 2.1.2 get NE spans by spaCy. They are flat
                # store the start word index, end word index (excluded) and the text of the NE spans.
                # i.e., (start_word_idx, end_word_idx, span_text)
                # The method is included in the spaCy package.
                # refer to https://spacy.io/usage/linguistic-features#named-entities
                # and https://spacy.io/api/span
                spacy_result += [(ent.start, ent.end, ent.text) for ent in spa_doc.ents]
                spacy_result = set(spacy_result)  # remove duplicates

                # 2.2 stanza
                # 2.2.1 get NP spans by stanza
                # refer to https://stanfordnlp.github.io/stanza/constituency.html
                # Here, Constituency parser of Stanza will split compound words formed by hyphen (-) into several words
                # e.g., 'United-States' will be split into 'United', '-' and 'States'
                constituency_string = sta_doc.sentences[0].constituency  # constituency parse tree (String) of the sentence

                # Convert string to nltk.tree.Tree
                # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.fromstring
                constituency_tree = Tree.fromstring(repr(constituency_string))

                # filter out all the NP subtrees
                # We can use a filter function to restrict the Tree.subtrees we want,
                # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.subtrees
                stanza_spans = dict()

                # Subtree.leaves() return a list of words in the subtree, e.g. ['the', 'United', 'States']
                # refer to https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.leaves
                subtrees = [subtree.leaves() for subtree in constituency_tree.subtrees(lambda t: t.label() == 'NP')]

                # However, as mentioned before, the compound words formed by hyphen (-) will be split into several
                # words by stanza. So we need to combine them back into a single word.
                # e.g., ['United', '-', 'States'] -> ['United-States']
                new_subtrees = self.merge_compound_words(subtrees)

                # We initiate the start character index and end character index with -1.
                # And the data format is a dict, where the key is the text of the NP span (to remove duplicates),
                # and the value is a tuple. i.e., {span_text: (start_character_idx, end_character_idx)}
                for n_subtree in new_subtrees:
                    stanza_spans.update({n_subtree: (-1, -1)})

                # 2.2.2 get NE spans by stanza
                # store the start character index, end character index (excluded) and the text of the NE spans.
                # i.e., (start_character_idx, end_character_idx, span_text)
                # The method is included in the stanza package.
                # refer to https://stanfordnlp.github.io/stanza/ner.html
                # and https://stanfordnlp.github.io/stanza/data_objects.html#span
                for ent in sta_doc.ents:
                    stanza_spans.update({ent.text: (ent.start_char, ent.end_char)})

                # store the start index, end index (excluded) and the text of the NP/NE span.
                stanza_result = []
                for span_text, (start_ch_idx, end_ch_idx) in stanza_spans.items():
                    # stanza will replace some special characters with special tokens, e.g., '(' -> '-LRB-', '[' -> '-LSB-'
                    # We need to replace them back to the original characters.
                    # But we need to use escape character in the regular expression.
                    span_text = self.replace_special_tokens(span_text)

                    if start_ch_idx == -1 and end_ch_idx == -1:  # -1 means the start/end character index is not available
                        # Find the start character index and end character index of the first matched NP/NE span.
                        match = sent.find(span_text)
                        if match == -1:  # no matched NP/NE span
                            continue
                        else:
                            start_ch_idx, end_ch_idx = match, match + len(span_text)

                    # To get the start position of the first word of the matched NP span,
                    # we just need to count the number of spaces before the start character
                    start = sent[ :start_ch_idx].count(' ')

                    # To get the end position of the last word of the matched NP span,
                    # we just need to count the number of spaces before the end character
                    end = sent[ :end_ch_idx].count(' ') + 1  # end position of the NP span, excluded
                    stanza_result.append((start, end, span_text))

                stanza_result = set(stanza_result)  # remove duplicates

                # 2.3. Select the union of two parsers' recognition results (NP/NE spans)
                # convert start/end index to string, to be consistent with the format of spans. This operation ensures
                # that the tuple is successfully converted to pyarrow and then serialized into a JSON/JSONL array
                max_span_len = kwargs['span_portion'] * len(sent)
                if kwargs['mode'] == 'strict':
                    # In strict (default) mode, we get spans based on intersection of spaCy and Stanza results.
                    spans = [(str(start), str(end), text)
                             for start, end, text in list(spacy_result & stanza_result)
                             if len(text) <= max_span_len  # filter out long span
                            ]
                else:
                    # In loose mode, we get spans based on union of spaCy and Stanza results.
                    spans = [(str(start), str(end), text)
                             for start, end, text in list(spacy_result | stanza_result)
                             if len(text) <= max_span_len  # filter out long span
                            ]
                if len(spans) > 0:  # filter out empty np_span
                    out_sentences.append(sent)
                    out_spans.append(spans)

        return {
            'sentence': out_sentences,
            'span': out_spans,
        }

    def pre_process(self):
        """
        Pre-process the data, including:
            1) get raw sentences (sentence only).
            2) remove duplicates.
            3) merge all datasets into a single file
        :return:
        """
        print("="*20 + " Pre-processing " + "="*20)

        # 1. init
        preprocess_cfg = self.config['preprocess']
        in_dirs = preprocess_cfg['in_dirs']
        out_dir = preprocess_cfg['out_dir']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, 'datasets.json')
        if os.path.exists(out_file):
            print(f'{out_file} already exists!')
            return

        # 2. start pre-processing
        unique_sents = set()
        with jsonlines.open(out_file, 'w') as writer:
            for in_dir in in_dirs:
                for path, dir_lst, file_lst in os.walk(in_dir):
                    for file_name in file_lst:
                        assert file_name.endswith('.json') or file_name.endswith('.jsonl'), \
                            f'File format not supported. Only json and jsonl are supported.'

                        in_file = os.path.join(path, file_name)
                        print(f'Pre-processing {in_file}, output to {out_file}')
                        # get raw sentences and remove duplicates
                        with jsonlines.open(in_file) as reader:
                            for obj in tqdm(reader, desc='reading'):
                                left_context_token = obj['left_context_token']
                                mention_span = obj['mention_span']
                                right_context_token = obj['right_context_token']
                                raw_sent = ' '.join(left_context_token + mention_span.split(' ') + right_context_token)
                                unique_sents.add(raw_sent)
            for sent in tqdm(unique_sents, desc='writing'):
                writer.write({'sentence': sent})
        print("=" * 20 + " Pre-processing Done " + "=" * 20)

    def process(self, stage):
        """
        Process the data in the given stage.

        :param stage: The stage to be processed.
        :return:
        """
        # set 'spawn' start method in the main process
        # refer to https://huggingface.co/docs/datasets/process#map
        multiprocess.set_start_method('spawn')

        # 1. init stage config to get fn_kwargs
        stage_cfg = self.config[stage]
        in_dir = stage_cfg['in_dir']
        out_dir = os.path.join(self.data_dir, stage)
        out_dir = os.path.join(out_dir, stage_cfg['mode'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if stage == 'stage1':
            process_func = self.stage1 # Specify the function to be used in the given stage.

        # 2. process each file in the given directory
        for path, dir_lst, file_lst in os.walk(in_dir):
            for file_name in file_lst:
                assert file_name.endswith('.json') or file_name.endswith('.jsonl'), \
                    f'File format not supported. Only json and jsonl are supported.'

                # 2.1. get input and output file path
                in_file = os.path.join(path, file_name)
                out_file = os.path.join(out_dir, file_name)

                # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.shard
                # A dataset without a loading script by default loads all the data into the train split
                # So we need to specify the 'train' split to process it.
                # refer to https://huggingface.co/docs/datasets/loading#hugging-face-hub
                dataset = load_dataset('json', data_files=in_file)
                dataset = dataset['train']

                # 2.2. shard the large dataset into num_shards pieces, and process one piece at a time
                processed_datasets = []
                for idx in range(self.num_shards):
                    print("="*20  + f'Processing shard {idx+1}/{self.num_shards} ' + "="*20)
                    sub_dataset = dataset.shard(num_shards = self.num_shards, index = idx, contiguous=True)

                    # 2.3. process the dataset
                    # https://huggingface.co/docs/datasets/process#map
                    # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.map
                    sub_dataset = sub_dataset.map(process_func,
                                          fn_kwargs = stage_cfg,  # kwargs for process_func
                                          batched = True,
                                          with_rank = True,
                                          batch_size = stage_cfg['batch_size_per_device'] * stage_cfg['batch_num_per_device'],
                                          num_proc = self.num_workers,
                                          remove_columns = dataset.column_names,  # Remove unnecessary columns from the original dataset
                                          )
                    processed_datasets.append(sub_dataset)

                # 3. save the processed dataset
                dataset = concatenate_datasets(processed_datasets)  # merge all shards into a single dataset
                dataset.to_json(out_file)

    def statistic(self):
        """
        statistic dataset information.
        :return:
        """
        pass

def main():
    config_file = r'config.yml'
    processor = Processor(config_file)
    processor.pre_process()  # pre-process the data
    processor.process('stage1')  # stage1, filter short sentences, get NP spans and NE spans in sentences. English only.

if __name__ == '__main__':
    main()
