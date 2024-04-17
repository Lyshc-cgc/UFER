import re
import json
import torch
import spacy
import stanza
import os
import multiprocess
import jsonlines
import shutil
import copy
import wandb
import numpy as np

from datasets import Dataset
from vllm import LLM, SamplingParams
from nltk.tree import Tree
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from util.type_util import TypeWordv2, get_type_word, del_none, get_best_definition, disambiguate_type_word
from util.func_util import get_config, batched, eval_anno_quality

class Processor:
    """
    Class Processor used to process the UFER dataset including:
        1) get type information
        2) pre-process the dataset
        3) process the dataset in each stage
    """
    def __init__(self, config_file):
        self.config = get_config(config_file)
        self.cuda_devices = self.config['cuda_devices']
        self.data_dir = self.config['data_dir']

        # store type information
        self.type_num = 0  # total number of type words
        self.type_words = []  # a list to store type words, just their literal values
        self.type_info = dict()  # a dict to store all type information. key: type word, value: TypeWordv2 object
        self.type2id = dict()  # a dict to store the mapping from type word to its id
        self.id2type = dict()  # a dict to store the mapping from id to type word

    @staticmethod
    def _merge_compound_words(sents: list[list[str]]) -> list[str]:
        """
        Used in stage1.
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
    def _modify_spacy_tokenizer(nlp):
        """
        Used in stage1.
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
                    r"(?<=[{a}])[:<>=/](?=[{a}])".format(a=ALPHA),  # Modified regex to only split words on '/' if it is preceded by a character
                ]
        )
        infix_re = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer
        return nlp

    @staticmethod
    def _replace_special_tokens(sent: str) -> str:
        """
        Used in stage1.
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

    def build_type_info(self, cache_stage=0):
        """
        To build type information of the UFER dataset, we need to do 4 things.

        First, get type word (including its definition) according to the original type words. The result will be stored in the 'cache_info_file_0' file.

        Second, we should manually :
            1) (carefully) For those words that cannot be found in the dictionary api and Wikipedia,
                we need to search the definitions of those words from other sources and  update the 'cache_info_file_0' file.
                Some words may not have a suitable definition, and we label their definitions as 'None'.
            2) (roughly) Check for words that is not suitable as a category, and we label their definitions as 'None'.
            3) (roughly) Check for redundant words (i.e., words with similar definition already exist), and we label
                redundant words' definitions as 'None'.

        Third, remove the words with the definition 'None'. The result will be stored in the 'cache_info_file_1' file.

        Forth, get the best definition of each type word. The result will be stored in the 'cache_info_file_2' file.

        (deprecated) Fifth, remove similar meanings using Clustering and LLMs. The result will be stored in the 'cache_info_file_3' file.
        :param cache_stage: The stage of the cache file. We use this param to control the building process.
            1) 0: no cache file, we should build from scratch;
            2) 1: cache file with 'None' removed, skip the first step;
            3) 2: cache file after getting the best definition of each type word, skip the first two steps.
            4) 3: ((deprecated)) cache file after meaning disambiguation, skip the first three steps
        """
        assert cache_stage in [0, 1, 2], f"cache_stage must be one of (0, 1, 2)!"
        type_cfg = get_config(self.config['type'])  # get type configuration
        work_dir = type_cfg['work_dir']  # get the work directory
        original_type_words = os.path.join(work_dir, type_cfg['original_type_words'])

        # 1. First, get type word (including its definition) according to the original type words
        cache_info_file_0 = os.path.join(work_dir, type_cfg['cache_info_file_0'])
        if cache_stage == 0 and not os.path.exists(cache_info_file_0):
            print("=" * 20 + " Building type information of the UFER dataset " + "=" * 20)

            # Due to some reason (network or request limit), the build process stop.
            # We can skip those type words whose information have been build and continue from the next new type word.
            # If start_id = 0, we still can build type information of the UFER dataset from scratch
            start_id = 0
            if os.path.exists(cache_info_file_0):
                with open(cache_info_file_0, 'r') as reader:
                    lines = reader.readlines()
                    start_id = len(lines)
                    print(f'{start_id} type words have been gotten. We continue build process from the {start_id+1}-th type word')

            with open(original_type_words, 'r') as reader, jsonlines.open(cache_info_file_0, 'a') as writer:
                for idx, line in enumerate(tqdm(reader.readlines(), desc='building type info...')):
                    if idx < start_id:  # Skip type information that has been established
                        continue
                    word = line.strip()
                    type_word_obj = get_type_word(word, idx, **type_cfg)
                    writer.write(type_word_obj.__dict__)
            cache_stage += 1

        # 2. Second, do some manual work on the cache info file.
        # 2.1 (carefully)
        # For those words that cannot be found in the dictionary api and Wikipedia,
        # we need to search the definitions of those words from other sources and  update the 'cache_info.jsonl' file.
        # Some words may not have a suitable definition, and we label their definitions as 'None'.
        # 2.2 (roughly)
        # Check for words that is not suitable as a category, and we label their definitions as 'None'.
        # 2.3 (roughly)
        # Check for redundant words (i.e., words with similar definition already exist),
        # and we label redundant words' definitions as 'None'.
        # ...

        cache_info_file_1 = os.path.join(work_dir, type_cfg['cache_info_file_1'])
        if cache_stage == 1:
            # 3. Third, remove the words with the definition 'None'.
            print("Remove the words with the definition 'None'.")
            del_none(in_file=cache_info_file_0, out_file=cache_info_file_1)
            cache_stage += 1

        cache_info_file_2 = os.path.join(work_dir, type_cfg['cache_info_file_2'])
        if cache_stage == 2:
            # 4. Forth, get the best definition of each type word.
            print("Get the best definition of each type word.")
            get_best_definition(in_file=cache_info_file_1,
                                out_file=cache_info_file_2,
                                cuda_devices=self.cuda_devices,
                                **type_cfg)
            cache_stage += 1

        type_info_file = os.path.join(work_dir, type_cfg['type_info_file'])
        shutil.copyfile(cache_info_file_2, type_info_file)
        # Step 5 is not necessary, depraecated
        # cache_info_file_3 = os.path.join(work_dir, type_cfg['cache_info_file_3'])
        # if cache_stage == 3:
        #     # 5. Forth, remove similar meanings using Clustering and LLMs.
        #     print("Remove similar meanings using Clustering and LLMs.")
        #     disambiguate_type_word(in_file=cache_info_file_2,
        #                            out_file=cache_info_file_3,
        #                            cuda_devices= self.cuda_devices,
        #                            **type_cfg)  # type_cfg is the type configuration
        #     print("Building type information of the UFER dataset Done!")
        # shutil.copyfile(cache_info_file_3, type_info_file)
        return cache_info_file_2

    def get_type_info(self, type_info_file=None):
        """
        Get type information of the UFER dataset from the type_info_file.

        :param type_info_file: The file to store type information. If None, we will use the default file in the config file.
        """
        type_cfg = get_config(self.config['type'])

        # write the type words to the type_words file
        type_words_file = os.path.join(type_cfg['work_dir'], type_cfg['type_words_file'])
        if type_info_file is None:
            type_info_file = type_cfg['type_info_file']
        with jsonlines.open(type_info_file,'r') as reader, open(type_words_file, 'w') as f:
            for line in tqdm(reader, desc='get type info...'):
                type_word_obj = TypeWordv2(**line)
                self.type_num += 1
                self.type_words.append(type_word_obj.word)
                self.type_info.update({type_word_obj.word: type_word_obj})
                self.id2type.update({type_word_obj.id: type_word_obj.word})
                self.type2id.update({type_word_obj.word: type_word_obj.id})
                f.write(type_word_obj.word + '\n')
        print('Get type information done!')

    def _prep_cand_type_words_for_stage2(self, candidate_type_words_nums=10, candidate_type_words_template=None, with_def=True):
        """
        Used before stage2.
        Prepare the candidate type words input to prompts.

        :param candidate_type_words_nums: The number of candidate type words with definitions input to prompts
        :param candidate_type_words_template: The template for candidate type words
        :param with_def: Whether to include the definition of the type word
        :return:
        """
        # a Sequence to store all type information.
        # Each element is dict in a form of {'id': id of this element, 'word': type word, 'definition': definition of the type word}
        candidate_type_words = []
        for idx, type_word_obj in enumerate(self.type_info.values()):
            candidate_type_words.append({'id': idx, 'word': type_word_obj.word, 'definition': type_word_obj.definition})

        batch_cand_type_words = []
        for batch_cad_ty_words in batched(candidate_type_words, candidate_type_words_nums):
            if with_def:  # with definitions
                # [{'id': 0, 'word': x0, 'definition': xx'}, {'id': 1, 'word': x1, 'definition': xx'}]
                # -> '<id>: 0, <word>: x0, <definition>: xx <id>: 1, <word>: x1, <definition>: xx'
                tmp = [candidate_type_words_template['wd'].format(id=e['id'], word=e['word'], definition=e['definition']) for e in batch_cad_ty_words]
            else:  # without definitions
                # [{'id': 0, 'word': x0}, {'id': 1, 'word': x1}] -> '<id>: 0, <word>: x0 <id>: 1, <word>: x1'
                tmp = [candidate_type_words_template['wod'].format(id=e['id'], word=e['word']) for e in batch_cad_ty_words]
            batch_cand_type_words.append('\n'.join(tmp))
        return candidate_type_words, batch_cand_type_words

    @staticmethod
    def _init_chat_message_for_stage2(anno_model_name: str, **anno_model_cfg) -> list[None|dict[str,str]]:
        """
        Used before stage2.
        Init the chat messages for the annotation models.

        :param anno_model_name: The name of the annotation model used in the stage2.
        :param anno_model_cfg: The parameters of the annotation model.
        :return:
        """
        if anno_model_name == 'Qwen':
            # for Qwen and Yi-ft, we need to input the system's prompt and the user's prompt
            # https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GPTQ-Int8
            # https://huggingface.co/TheBloke/nontoxic-bagel-34b-v0.2-GPTQ
            chat_message = [
                {"role": "system", "content": anno_model_cfg['anno_sys_prompt_batch']},
            ]
            for example in anno_model_cfg['anno_examples_batch']:
                usr_content = anno_model_cfg['anno_usr_prompt_batch'].format(sentence=example['sentence'],
                                                                                candidate_type_words=example['candidate_type_words'])
                chat_message.append({"role": "user", "content": usr_content})
                chat_message.append({"role": "assistant", "content": example['output']})
        else:
            # e.g., for mistral and Yi, we only need to input the user's prompt
            chat_message = []
        return chat_message

    @staticmethod
    def _eval_anno_quality(instances, sub_field):
        """
        Evaluate the quality of annotations.

        :param instances: Dict[str, List], a list of instances.
        :param sub_field: the subject filed in instances
        :return:
        """
        pass


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
        preprocess_cfg = get_config(self.config['preprocess'])
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

    def post_process(self):
        pass

    def stage1(self, instances, rank = 0, **kwargs):
        """
        Stage1, filter short sentences, get noun phrase (NP) spans and named entity (NE) spans in sentences. English only.
        In this stage, we get flat spans by spaCy. Meanwhile, we get nested spans by spaCy and Stanza.

        :param instances: Dict[str, List], A batch of instances.
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

        # 0. GPU setting for stage1
        if self.cuda_devices == 'all':
            # set the GPU can be used by stanza in this process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

            # specify the GPU to be used by spaCy, which should be same as above
            # https://spacy.io/api/top-level#spacy.prefer_gpu
            spacy.prefer_gpu(rank % torch.cuda.device_count())
        else:
            cuda_devices = str(self.cuda_devices).split(',')
            gpu_id = rank % len(cuda_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[gpu_id])

            # specify the GPU to be used by spaCy, which should be same as above
            spacy.prefer_gpu(int(cuda_devices[gpu_id]))

        # load a spaCy and a stanza model in each process
        spacy_nlp = spacy.load(name = kwargs['spacy_model']['name'])
        spacy_nlp = self._modify_spacy_tokenizer(spacy_nlp)  # modify the spaCy tokenizer
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
            batch_texts = self._merge_compound_words(batch_texts)

            # 1.4. replace special tokens with original characters
            batch_texts = [self._replace_special_tokens(sent) for sent in batch_texts]

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
                new_subtrees = self._merge_compound_words(subtrees)

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
                    span_text = self._replace_special_tokens(span_text)

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

                assert kwargs['mode'] in ['strict', 'loose'], f"mode must be one of ('stric', loose)!"
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

    def stage2(self, instances, **kwargs):
        """
        Stage2, annotate the given entity mention  in the instances.

        :param instances: Dict[str, List], A batch of instances. In stage2, it is a dataset shard.
        :param kwargs:
            1) shard_idx: The index of the shard.
            2) out_dir: The output directory.
            3) candidate_type_words: the candidate type words with definitions. Each element is a dict in a form of '{id: x, word: xx, definition: xxx}'
            4) batch_cand_type_words: the batched candidate type words. Each element is a string composed of a batch of candidate type words.
            5) chat_message_templates: the chat message template for each annotating model
            6) with_def: whether to include definitions of type words.
            7) anno_models_wd: annotation model configs to include definitions of type words.
            8) anno_models_wod: annotation model configs to exclude definitions of type words.
        :return:
        """
        # init wandb
        wandb.init(
            project='UFER',
            config=kwargs
        )

        # 0. GPU setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
        # set GPU device
        if self.cuda_devices == 'all':
            # set the GPU can be used
            cuda_devices = [str(i) for i in range(torch.cuda.device_count())]
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

        def _generate_chat_messages(instances, anno_model_name):
            """
            Generate chat messages for each instance. Meanwhile, init the labels for each instance.

            :param instances: The instances to be annotated.
            :param anno_model_name: The name of the annotating model.
            :return:
            """
            for sent_id, (sentence, span) in enumerate(zip(instances['sentence'], instances['span'])):
                # span is the NP/NE spans in the sentence
                # e.g. [["14","17","a big ship"],["0","2","Emerald Princess"],["13","14","she"],["2","5","The Emerald Princess"],["7","10","Saturday 5th May"]]
                instances[f'{anno_model_name}_labels'].append([])  # init labels annotated by each annotating model
                instances[f'{anno_model_name}_label_ids'].append([])  # init label ids annotated by each annotating model
                # instances[f'{anno_model_name}_anwsers'].append([])  # init answer output by each annotating model

                for span_id, (start, end, entity_mention) in enumerate(tuple(span)):
                    start, end = int(start), int(end)
                    sent = sentence.split(' ')
                    sentence = ' '.join(sent[:start] + ['[e]', entity_mention, '[/e]'] + sent[end:])
                    instances[f'{anno_model_name}_labels'][sent_id].append(set())  # init labels for each entity mention
                    instances[f'{anno_model_name}_label_ids'][sent_id].append(set())  # init label ids for each entity mention
                    # instances[f'{anno_model_name}_anwsers'][sent_id].append(set())  # init answer output for each entity mention

                    for b_c_t_words in kwargs['batch_cand_type_words']:
                        chat_message = copy.deepcopy(kwargs['chat_message_templates'][anno_model_name])
                        usr_content = anno_model_cfg['anno_usr_prompt_batch'].format(sentence=sentence, candidate_type_words=b_c_t_words)
                        chat_message.append({"role": "user", "content": usr_content})
                        yield sent_id, span_id, chat_message  # yield chat message, the ID of the sentences it contains and the ID of the entity mention it contains

        # 1. annotate entity type by multiple LLMs.
        anno_model_cfgs = kwargs['anno_models_wd'] if kwargs['with_def'] else kwargs['anno_models_wod']
        for anno_model_cfg in anno_model_cfgs:
            anno_model_name = anno_model_cfg['name']
            instances.update({f'{anno_model_name}_labels': []})  # store type words annotated by each annotating model
            instances.update({f'{anno_model_name}_label_ids': []})  # store type word ids  annotated by each annotating model
            # instances.update({f'{anno_model_name}_anwsers': []})  # store the answer output by each annotating model
            print("=" * 20 + f'Start to be annotated by {anno_model_name} ' + "=" * 20)

            # 1.1 Check if there is a cached annotation result file
            out_dir = kwargs['out_dir']  # output directory
            shard_idx = kwargs['shard_idx']  # the id of the dataset shard being processed
            shard_out_file = os.path.join(out_dir, f'{anno_model_name}/datasets_shard_{shard_idx}.jsonl')
            if os.path.exists(shard_out_file):
                print(f'{shard_out_file} already exists! Skip annotate dataset shard {shard_idx} by {anno_model_name}')
                continue

            # 1.2 Import the annotating model
            # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
            anno_model = LLM(model=anno_model_cfg['checkpoint'],
                             tensor_parallel_size=gpu_num,
                             dtype=anno_model_cfg['dtype'],
                             gpu_memory_utilization=anno_model_cfg['gpu_memory_utilization'],
                             trust_remote_code=True)
            sampling_params = SamplingParams(temperature=anno_model_cfg['anno_temperature'],
                                             top_p=anno_model_cfg['anno_top_p'],
                                             max_tokens=anno_model_cfg['anno_max_tokens'],
                                             repetition_penalty=anno_model_cfg['repetition_penalty'])

            # get anno_model's tokenizer to apply the chat template
            # https://github.com/vllm-project/vllm/issues/3119
            anno_tokenizer = anno_model.llm_engine.tokenizer.tokenizer

            # 1.3 batch process
            num_shards = kwargs['num_shards']
            pbar = tqdm(batched(_generate_chat_messages(instances, anno_model_name), anno_model_cfg['anno_bs']),
                        desc=f'annotating shards {shard_idx}/{num_shards}')
            num_spans = 0  # store the number of spans in the dataset shard
            json_pattern = r'\{\{(.*?)\}\}'  # the pattern to extract JSON string
            for batch in pbar:  # batch is a tuple like ((sent_id_0, span_id_0, chat_0),(sent_id_1, span_id_1, chat_1)...)
                batch_sent_ids, batch_span_ids, batch_chats = [], [], []
                for sent_id, span_id, chat in batch:
                    batch_sent_ids.append(sent_id)
                    batch_span_ids.append(span_id)
                    batch_chats.append(chat)
                num_spans += len(batch_span_ids)
                # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                templated_batch_chats = anno_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
                outputs = anno_model.generate(templated_batch_chats, sampling_params)  # annotate
                # for test
                test_answer = []
                for output in outputs:
                    test_answer.append({'prompt': output.prompt, 'output': output.outputs[0].text})
                for sent_id, span_id, output in zip(batch_sent_ids, batch_span_ids, outputs):
                    # extract JSON string from output.outputs[0].text
                    # out_answer is a string like '2782, 2783, 2788'
                    result = re.findall(json_pattern, output.outputs[0].text, re.DOTALL)
                    if result and len(result) >= 1:  # only extract the first JSON string
                        try:
                            # todo, json.loads() is not safe, we should use a safer way to extract JSON string or use try catch
                            out_answer = json.loads('{' + result[0]  + '}')['answer']
                            out_answer = out_answer.strip().split(',')
                            out_answer = [e.strip() for e in out_answer]
                        except json.JSONDecodeError:
                            out_answer = ['None']  # we assign 'None' to out_answer if we cannot extract the JSON string, so that we can continue the loop
                    else:
                        out_answer = ['None']  # we assign'None' to out_answer if we cannot extract the JSON string, so that we can continue the loop

                    # check the answers
                    if len(out_answer) == 1 and out_answer[0] == 'None':
                        # if the model cannot find a suitable type word in this batch of candidate type words,
                        # we should continue the loop
                        continue

                    labels, label_ids = [], []
                    for e in out_answer:
                        type_word = kwargs['candidate_type_words'][int(e)]
                        labels.append(type_word['word'])
                        label_ids.append(self.type2id[type_word['word']])
                    instances[f'{anno_model_name}_labels'][sent_id][span_id].update(labels)
                    instances[f'{anno_model_name}_label_ids'][sent_id][span_id].update(label_ids)
                    # instances[f'{anno_model_name}_anwsers'][sent_id][span_id].update(out_answer)
                break

            # 1.4 cache the annotation result of each annotating model
            cached_instances = {
                'sentence': instances['sentence'],
                'span': instances['span'],
                'labels': instances[f'{anno_model_name}_labels'],
                'label_ids': instances[f'{anno_model_name}_label_ids'],
                # 'anwsers': instances[f'{anno_model_name}_anwsers'],
            }
            Dataset.from_dict(cached_instances).to_json(shard_out_file)

        # 2. evaluate the annotation quality of this shard
        # 2.1 prepare the evaluation data containing category assignment with subjects in rows and annotators in columns.
        # https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.aggregate_raters.html#
        # For evaluation on multi-label annotation, we cast multi-label (N types) into N binary-classification for each span.
        # And then, calculate the inter-annotator agreement (IAA). i.e., Fleiss' Kappa, Krippendorff's Alpha, etc.
        num_anno_models = len(anno_model_cfgs)
        eval_data = np.zeros((num_spans * self.type_num, num_anno_models), dtype=np.int8)
        span_id = -1  # the ID of the span
        for model_id, anno_model_cfg in enumerate(anno_model_cfgs):
            anno_model_name = anno_model_cfg['name']
            for sent_spans_labels in instances[f'{anno_model_name}_label_ids']:  # e.g., sent_spans_labels=[[1,2], [3,4], [5,6]]
                span_id += 1
                for span_labels in sent_spans_labels:  # e.g., span_labels=[1,2]
                    for label in span_labels:  # e.g., label=1
                        eval_data[span_id * self.type_num + label, model_id] = 1

        # 2.2 evaluate the annotation quality
        qual_res = eval_anno_quality(eval_data, metric='fleiss_kappa')
        wandb.log(qual_res)
        return instances

    def process(self, stage):
        """
        Process the data in the given stage.

        :param stage: The stage to be processed.
        :return:
        """
        # set 'spawn' start method in the main process
        # refer to https://huggingface.co/docs/datasets/process#map
        print("=" * 20 + f" Processing {stage} " + "=" * 20)
        multiprocess.set_start_method('spawn')

        # 1. init stage config
        stage_cfg = get_config(self.config[stage])
        in_dir = stage_cfg['in_dir']
        out_dir = os.path.join(self.data_dir, stage)
        out_dir = os.path.join(out_dir, stage_cfg['mode'])
        if stage == 'stage1':
            process_func = self.stage1 # Specify the function to be used in the given stage.
            batch_size = stage_cfg['batch_size_per_device'] * stage_cfg['batch_num_per_device']
            if not os.path.exists(out_dir):  # out_dir/mode/
                os.makedirs(out_dir)

        elif stage == 'stage2':
            process_func = self.stage2
            batch_size = None  # we do not need to specify batch_size in stage2, just input the whole dataset shards.
            in_dir = os.path.join(in_dir, stage_cfg['mode'])  # in stage2, we need to specify the input mode
            stage_cfg.update({'out_dir': out_dir})  # in stage2, we need to specify the output directory

            # prepare the candidate type words input to prompts
            candidate_type_words, batch_cand_type_words = self._prep_cand_type_words_for_stage2(stage_cfg['candidate_type_words_nums'],
                                                                                     stage_cfg['candidate_type_words_template'],
                                                                                                with_def=stage_cfg['with_def'])
            stage_cfg.update({'candidate_type_words': candidate_type_words})
            stage_cfg.update({'batch_cand_type_words': batch_cand_type_words})

            # init output dir and the chat messages for each annotation model
            stage_cfg.update({'chat_message_templates': dict()})  # store the chat message templates for the annotation models
            anno_model_cfgs = stage_cfg['anno_models_wd'] if stage_cfg['with_def'] else stage_cfg['anno_models_wod']
            for anno_model_cfg in anno_model_cfgs:
                anno_model_name = anno_model_cfg['name']
                out_dir_model = os.path.join(out_dir, anno_model_name)  # out_dir/mode/annt_model_name/
                if not os.path.exists(out_dir_model):
                    os.makedirs(out_dir_model)
                stage_cfg['chat_message_templates'][anno_model_name] = self._init_chat_message_for_stage2(anno_model_name, **anno_model_cfg)

        # 2. process each file in the given directory
        for path, dir_lst, file_lst in os.walk(in_dir):
            for file_name in file_lst:
                assert file_name.endswith('.json') or file_name.endswith('.jsonl'), \
                    f'File format not supported. Only json and jsonl are supported.'

                # 2.1. get input and output file path
                in_file = os.path.join(path, file_name)
                out_file = os.path.join(out_dir, file_name)
                if os.path.exists(out_file):
                    print(f'{out_file} already exists!')
                    continue

                # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.shard
                # A dataset without a loading script by default loads all the data into the train split
                # So we need to specify the 'train' split to process it.
                # refer to https://huggingface.co/docs/datasets/loading#hugging-face-hub
                dataset = load_dataset('json', data_files=in_file)
                dataset = dataset['train']

                # 2.2. shard the large dataset into num_shards pieces, and process one piece at a time
                # https://huggingface.co/docs/datasets/process#shard
                processed_datasets = []
                num_shards = stage_cfg['num_shards']
                for shard_idx in range(num_shards):
                    print("=" * 20 + f'Processing shard {shard_idx}/{num_shards} ' + "=" * 20)
                    sub_dataset = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
                    stage_cfg.update({'shard_idx': shard_idx})  # update the shard index
                    # 2.3. process the dataset
                    # https://huggingface.co/docs/datasets/process#map
                    # https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.map
                    sub_dataset = sub_dataset.map(process_func,
                                                  fn_kwargs = stage_cfg,  # kwargs for process_func
                                                  batched = True,
                                                  with_rank = stage_cfg['with_rank'],
                                                  batch_size = batch_size,
                                                  num_proc = stage_cfg['num_workers'],
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
    # 0. init config
    config_file = r'config.yml'
    processor = Processor(config_file)

    # 1. pre-process the data
    # processor.pre_process()

    # 2. stage1, filter short sentences, get NP spans and NE spans in sentences. English only.
    # processor.process('stage1')

    # 3. build type information first
    type_cfg = get_config(processor.config['type'])
    type_work_dir = type_cfg['work_dir']
    type_info_file = os.path.join(type_work_dir, type_cfg['type_info_file'])
    if not os.path.exists(type_info_file):
        type_info_file = processor.build_type_info(cache_stage=2)  # return the type information after disambiguation
    processor.get_type_info(type_info_file)

    # 4. stage2, annotate the given entity mention in the instances by multiple LLMs.
    processor.process('stage2')

if __name__ == '__main__':
    main()
