import spacy
import stanza
import os
import yaml
import multiprocess
import torch
import re
from datasets import load_dataset
from yaml.loader import SafeLoader
from phrasetree.tree import Tree


class Processor:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=SafeLoader)
        self.lang = self.config['lang']
        assert self.lang == 'en', f'Language {self.lang} is not supported! Please use "en"'
        self.in_dir = self.config['in_dir']
        self.out_dir = self.config['out_dir']
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.num_workers = self.config['num_workers']
        self.batch_size = self.config['batch_size']

    @staticmethod
    def stage1(instances, rank, **kwargs):
        """
        Stage1, filter short sentences, get noun phrase (NP) spans and named entity (NE) spans in sentences. English only.

        :param instances: A batch of instances.
        :param rank: The rank of the current process.
        :param kwargs:
            1) min_len: The number of words in the sentence to be processed must exceed min_len. Default is 10.
            2) spacy_model: The spacy model config we use.
            3) stanza_model: The stanza model config we use.
        :return:
        """
        # set the GPU can be used by stanza in this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

        # specify the GPU to be used by spacy, which should be same as above
        spacy.prefer_gpu(rank % torch.cuda.device_count())

        # load a spacy and a stanza model in each process
        spa_nlp = spacy.load(**kwargs['spacy_model'])
        sta_nlp = stanza.Pipeline(**kwargs['stanza_model'], download_method=None)

        out_sentences = []
        out_spans = []
        original_labels = []
        for lc, m, rc, label in zip(instances['left_context_token'], instances['mention_span'],
                                    instances['right_context_token'], instances['y_str']):
            raw_sent = lc + [m] + rc
            if len(raw_sent) < kwargs['min_len']:  # filter short sentences
                continue
            raw_sent = (' '.join(raw_sent).replace('-LRB-', '(').replace('-RRB-', ')').
                        replace('-LSB-', '[').replace('-RSB-', ']'))
            out_sentences.append(raw_sent)
            original_labels.append(label)

        # 1. spacy
        # refer to
        # 1) https://spacy.io/usage/processing-pipelines#processing
        # 2) https://spacy.io/api/language#pipe
        spa_results = []
        for doc in spa_nlp.pipe(out_sentences):
            # store the start index, end index (excluded) and the text of the NP span.
            # refer to https://spacy.io/usage/linguistic-features#noun-chunks
            tmp_result = [(chunk.start, chunk.end, chunk.text) for chunk in doc.noun_chunks]

            # store the start index, end index (excluded) and the text of the NE span.
            tmp_result += [(ent.start, ent.end, ent.text) for ent in doc.ents]
            spa_results.append(list(set(tmp_result)))  # remove duplicates

        # 2. stanza
        # refer to https://stanfordnlp.github.io/stanza/getting_started.html#processing-multiple-documents
        sta_results = []
        sta_docs = sta_nlp.bulk_process(out_sentences)
        for doc in sta_docs:
            constituency_string = doc.sentences[0].constituency  # constituency parse tree (String) of the sentence
            constituency_tree = Tree.fromstring(repr(constituency_string))  # convert string to phrase.tree.Tree (like nltk.tree.Tree)

            # filter out all the NP subtrees
            # subtree.leaves() return a list of words in the subtree, e.g. ['the', 'United', 'States']
            tmp_result = [' '.join(subtree.leaves()) for subtree in constituency_tree.subtrees(lambda t: t.label() == 'NP')]
            tmp_result += [ent.text for ent in doc.ents]
            sta_results.append(list(set(tmp_result)))  # remove duplicates

        # 3. select the NP/NE spans that recognized by both parser
        for spa_res, sta_res in zip(spa_results, sta_results):
            # convert start/end index to string, to be consistent with the format of spans. This operation ensures that
            # the tuple is successfully converted to pyarrow and then serialized into a JSON/JSONL array
            spans = [(str(start), str(end), span0) for start, end, span0 in spa_res
                     for span1 in sta_res if span0 in span1 or span1 in span0]
            # equivalent to
            # for start, end, span0 in spa_res:
            #     for span1 in sta_res:
            #         if span0 in span1 or span1 in span0:
            #             spans.append([start, end, span0])

            out_spans.append(list(set(spans)))  # remove duplicates

        return {
            'sentence': out_sentences,
            'np_span': out_spans,
            'original_label': original_labels,
        }


    def process(self, stage):
        """
        Process the data in the given stage.

        :param stage: The stage to be processed.
        :return:
        """
        # set 'spawn' start method in the main process
        # refer to https://huggingface.co/docs/datasets/process#map
        multiprocess.set_start_method('spawn')
        # gpu_num = torch.cuda.device_count()

        # 1. init stage config to get fn_kwargs
        stage_cfg = self.config[stage]
        if stage == 'stage1':
            process_func = self.stage1 # Specify the function to be used in the given stage.

        # 2. process each file in the given directory
        for _, dir_lst, file_lst in os.walk(self.in_dir):
            for file_name in file_lst:
                assert file_name.endswith('.json') or file_name.endswith('.jsonl'), \
                    f'File format not supported. Only json and jsonl are supported.'
                in_file = os.path.join(self.in_dir, file_name)
                out_file = os.path.join(self.out_dir, file_name)

                # A dataset without a loading script by default loads all the data into the train split
                dataset = load_dataset('json', data_files=in_file)
                dataset = dataset.map(process_func,
                                      fn_kwargs = stage_cfg,  # kwargs for process_func
                                      batched = True,
                                      with_rank = True,
                                      batch_size = self.batch_size,
                                      num_proc = self.num_workers,
                                      remove_columns = dataset['train'].column_names,  # Remove unnecessary columns from the original dataset
                                      )

                # 3. save the processed dataset
                # A dataset without a loading script by default loads all the data into the train split.
                # So we need to specify the 'train' split to be saved.
                # refer to https://huggingface.co/docs/datasets/loading#hugging-face-hub
                dataset['train'].to_json(out_file)

def main():
    config_file = r'config.yml'
    processor = Processor(config_file)
    processor.process('stage1')

if __name__ == '__main__':
    main()
