"""
This module is used to help build the type information of the UFER dataset, including:
1) get definition for type words in the UFER dataset.
    a) Make request to a free dictionary api {https://github.com/meetDeveloper/freeDictionaryAPI}, which is backend
    with Wiktionary {https://en.wiktionary.org/wiki/Wiktionary:Main_Page}.
    b) Parse the response we need into specific object. We use noun meaning of a word as the definition of the word.
    c) If we cannot find the query word at the free dictionary api, we will try to search on Wikipedia.
    d) We use the first two sentences of Wikipedia Encyclopedia corresponding to that word as the definition.
2) remove unsuitable type words.
"""

import re
import os
import requests
import json
import jsonlines
import wikipedia as wp
import pandas as pd
import torch
import itertools
from torch.utils.data import DataLoader, Dataset
from vllm import LLM, SamplingParams
from sklearn.cluster import BisectingKMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm


@dataclass
class TypeWord:
    """
    Class TypeWord used to store the information (id and definition) of each type word in the UFER dataset.
    """
    id: int  # id of the type word
    word: str  # the type word
    source: str  # The source of the definitions, should be one of ('dic_api', 'wiki', 'other').
    definitions: List[str]  # definitions of the type word

def get_from_dictionary(word: str, **kwargs) -> Optional[List[str]]:
    """
    Do a 'get' request to the free dictionary api to get information of the given query word.
    We only get the definition of query words as a noun word.

    :param word: the query word
    :param kwargs: some other parameters including,
        1) lang_type: language type of the dictionary you want. Default is 'en' for English dictionary.
        2) api_url: the url of the dictionary api.
    :return: a list of definitions if the query word can be found at the free dictionary api, otherwise None.
    """
    # 1. make a request
    prefix = kwargs['api_url'] + kwargs['lang_type'] + '/'
    res = requests.get(prefix + word)
    if res.status_code != 200:  # the query word cannot be found at the free dictionary api.
        new_word = word.replace('_', ' ')  # e.g., living_thing -> living thing
        res = requests.get(prefix + new_word)
        if res.status_code != 200:
            new_word = word.replace('_', '-')  # e.g., living_thing -> living-thing
            res = requests.get(prefix + new_word)
            if res.status_code != 200:
                return None

    # 2. parse the response from the free dictionary api in order to get definitions of the query word.
    # see details of the response format at https://dictionaryapi.dev/
    # We only get the definition of query words as a noun word
    res = json.loads(res.content)[0]
    for meaning in res['meanings']:
        if meaning['partOfSpeech'] == 'noun':
            definitions = [e['definition'] for e in meaning['definitions']]
            return definitions

    # 3. If the API's response to this query word does not include a noun definition, return None.
    return None

def get_from_wikipedia(word: str, **kwargs) -> Optional[List[str]]:
    """
    Search on Wikipedia to get the definition of the given query word.
    We only use the first two sentences of Wikipedia Encyclopedia corresponding to that word as the definition.

    :param word: the query word
    :param kwargs: other parameters including,
        1) sentences: the number of sentences of the definition you want. Default is 2.
        2) auto_suggest: whether to automatically search for alternative suggestions if the query word is not found. Default is False.
    :return: a list of definitions if the query word can be found on Wikipedia, otherwise None.
    """
    # use the first two sentences of Wikipedia Encyclopedia corresponding to that word as the definition
    # see details at https://wikipedia.readthedocs.io/en/latest/quickstart.html
    try:
        res = wp.summary(word, sentences=kwargs['sentences'], auto_suggest=kwargs['auto_suggest'])
        res = ' '.join(res.split('\n')[:2])
        definitions = [res]
    except (wp.exceptions.DisambiguationError, wp.exceptions.PageError):
        try:
            new_word = word.replace('_', ' ')
            res = wp.summary(new_word, sentences=kwargs['sentences'], auto_suggest=kwargs['auto_suggest'])
            res = ' '.join(res.split('\n')[:2])
            definitions = [res]
        except (wp.exceptions.DisambiguationError, wp.exceptions.PageError):
            try:
                new_word = word.replace('_', '-')
                res = wp.summary(new_word, sentences=kwargs['sentences'], auto_suggest=kwargs['auto_suggest'])
                res = ' '.join(res.split('\n')[:2])
                definitions = [res]
            except (wp.exceptions.DisambiguationError, wp.exceptions.PageError):
                print(f"{word} is ambiguous or not found. You should manually handle it.")
                # if the query word is ambiguous, we just return None and handle it manually later
                definitions = None
    return definitions

def get_type_word(word: str, id: int = 0, **kwargs) -> TypeWord:
    """
    Get the information of the given query word from the free dictionary api or Wikipedia
    and parse it into a TypeWord object.

    :param word: the query word
    :param id: the id of the query word
    :param kwargs: other parameters including,
        1) lang_type: language type of the dictionary you want. Default is 'en' for English dictionary.
        2) api_url: the url of the dictionary api.
        3) sentences: the number of sentences of the definition you want. Default is 2.
        4) auto_suggest: whether to automatically search for alternative suggestions if the query word is not found. Default is False.
    :return: A TypeWord object containing the information of the query word.
    """
    # 1. try to get the definitions of the query word from the free dictionary api
    definitions = get_from_dictionary(word, **kwargs)
    source = 'dic_api'

    # 2. if we cannot find the query word at the free dictionary api, try to search on Wikipedia
    if not definitions:
        definitions = get_from_wikipedia(word, **kwargs)
        source = 'wiki'

    if definitions: # Luckily, we found definitions from the dictionary api or Wikipedia
        return TypeWord(id, word, source, definitions)

    # 3. if still cannot find the query word, return a TypeWord object with the definition 'None',
    # we should manually search the definitions from the other source like Collins,
    # then update the information in the type_info_file.
    return TypeWord(id, word, 'other', ['None'])

def del_none(in_file, out_file):
    """
    read the `in_file` and delete the type words with the definition 'None', output to the `out_file`.
    :param in_file: file input.
    :param out_file: file output
    :return:
    """
    count = 0
    with jsonlines.open(in_file, 'r') as reader, jsonlines.open(out_file, 'w') as writer:
        for obj in tqdm(reader):
            if obj['definitions'] == ['None']:
                continue
            if not obj['source'] in ('dic_api', 'wiki', 'other'):
                print(obj)
            else:
                obj['id'] = count
                writer.write(obj)
                count += 1


def judge_by_llm(clusters, input_type_info, **kwargs):
    """
    Judge the similarity of the definitions in each cluster by LLMs and remove the redundant definitions from each cluster.

    :param clusters: The clusters of the definitions of the type words. Each cluster is a list of dict, where each dict
        contains the id of the definition and the definition.
    :param input_type_info: The information of the type words. It's a dict, where the key is the id of the type word and
    :param kwargs:
        1) jud_model: the LLM used to judge. model_name or path.
        2) jud_prompt: prompt for the judge model.
        3) jud_temperature: temperature for the judge model. Default is 0.
        4) jud_top_p: top_p for the judge model. he smaller the value, the more deterministic the model output is.
        5) jud_max_tokens: The maximum number of tokens to generate.
        6) jud_bs: batch size for jud_model.
        7) tensor_parallel_size: The number of GPUs you want to use for running multi-GPU inference.
        8) dtype: The data type of the input tensor. https://docs.vllm.ai/en/latest/models/engine_args.html#cmdoption-dtype
        9) jud_res_cache_file: the file to cache the judge results.
    :return: The redundant definitions' ids i.
    """
    # 4. judge by LLMs. We use vLLM for faster inference
    # 4.1 Import the judge model
    # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    jud_model = LLM(model=kwargs['jud_model'], tensor_parallel_size=kwargs['tensor_parallel_size'], dtype=kwargs['dtype'])
    sampling_params = SamplingParams(temperature=kwargs['jud_temperature'], top_p=kwargs['jud_top_p'],
                                     max_tokens=kwargs['jud_max_tokens'])

    # 4.2 Remove definitions with duplicate meanings in each cluster by pairwise comparison
    # 4.2.1 prepare the prompts input to the judge model
    definition_pairs, prompts = [], []
    pair_num_of_clusters = []  # store the number of pairs of each cluster
    defi_As, defi_Bs = [], []  # cache the definition A and B in each pair
    for cluster in clusters:
        # get definition pair in this clustering
        # https://docs.python.org/3.8/library/itertools.html#itertools.combinations
        defi_pair = list(itertools.combinations(cluster, 2))
        definition_pairs += defi_pair
        pair_num_of_clusters.append(len(defi_pair))
        for defi_A, defi_B in defi_pair:  # process the prompts
            defi_As.append(defi_A)
            defi_Bs.append(defi_B)
            prompts.append(kwargs['jud_prompt'].format(first_definition=defi_A['definition'],
                                                       second_definition=defi_B['definition']))

    # 4.2.2 prepare the batched prompts input to the judge model, and then judge
    jud_results = []  # store the judge result
    pattern = r'\d+'  # pattern to match the number in the output
    prompts_loader = DataLoader(prompts, batch_size=kwargs['jud_bs'], shuffle=False)
    with tqdm(total=len(prompts), desc='judging') as t:
        for batch_id, batch_prompts in enumerate(prompts_loader):
            outputs = jud_model.generate(batch_prompts, sampling_params)  # judge
            for instance_id, output in enumerate(outputs):
                jud_result = output.outputs[0].text.strip()
                if jud_result := re.search(pattern, jud_result):
                    jud_result = int(jud_result.group())
                    assert jud_result in (1, 0), f'LLM generated wrong result: {jud_result}'
                else:
                    raise ValueError(f"LLM didn't generated answer we needed: {jud_result}")
                jud_results.append(jud_result)
            t.update(kwargs['jud_bs'])

    # 4.2.3 cache the judge results
    cache_jud_results = {'jud_result': jud_results, 'defi_A': defi_As, 'defi_B': defi_Bs}
    pd.DataFrame(cache_jud_results).to_csv(kwargs['jud_res_cache_file'], index_label='id')

    # 4.2.4 Remove similar defs in each cluster according to the judge results
    start_pos = 0
    redundant_id = []  # to store the id of the redundant definition
    for pair_num in pair_num_of_clusters:
        # get definition pairs and judge result in this cluster
        # pair_num is the number of pairs of each cluster
        defi_pairs_in_cluster = definition_pairs[start_pos:start_pos + pair_num]
        jud_result_in_cluster = jud_results[start_pos:start_pos + pair_num]

        # only leave pairs with similar definitions
        # https://docs.python.org/3.8/library/itertools.html#itertools.compress
        # e.g. [(A,B), (A,C), (A,D), (A,E), (B,C), (B,D), (B,E), (C,D), (C,E)] -> [(A, B), (A, D), (B, D), (C, E)]
        # It means that obj A, B, and D have similar meanings, which is the same applies to obj C and E.
        # The judge result is a list like [1, 0, 1, 0...], where 1 means that the definition are similar in that pair.
        similar_pairs = itertools.compress(defi_pairs_in_cluster, jud_result_in_cluster)

        # Store the id of the redundant definition
        # group by the ID (i.e, defi_id) of the first definition in a pair
        # https://docs.python.org/3.8/library/itertools.html#itertools.groupby
        # e.g. [(A, B), (A, D), (B, D), (C, E)] -> (A, B), (A, D) | (B, D) | (C, E)
        for key, group in itertools.groupby(similar_pairs, key=lambda x: x[0]['defi_id']):
            # e.g. (A, B), (A, D) -> (A, B, D)
            definition_set = set()  # A set to store similar definition
            for pair in group:
                definition_set.update(pair)

            # Compare the number of definitions of source words.
            # We leave the word with the least definitions.
            # And the others are considered as the redundant word
            # input_type_info like {1:{"word": "body_part", "source": "other", "definitions": ["a part of a human body."]}}
            # x['defi_id'] is the defi_id like 1,
            # input_type_info[x['defi_id']] is {"word": "body_part", "source": "other", "definitions": ["a part of a human body."]}
            # input_type_info[x['defi_id']]['definitions'] is ["a part of a human body."]
            tmp_redundant_id = [e['defi_id'] for e in definition_set]
            tmp_redundant_id.sort(key=lambda x: len(input_type_info[x['defi_id']]['definitions']))
            left_definition_id = tmp_redundant_id[0]
            tmp_redundant_id.remove(left_definition_id)  # the others are redundant
            redundant_id += tmp_redundant_id

    return redundant_id

def disambiguate_type_word(in_file, out_file, cuda_devices, **kwargs):
    """
    Disambiguate type word given in the `in_file` using Clustering and LLMs based on their definition.

    We divide definitions with similar meanings into the same cluster, and then use LLMs to remove redundant definitions
    from each cluster. This method has a lower computational cost than simply using LLM to remove redundant definitions.
    It's noting that all clusters should be about same size to reduce the computational costs of LLMs.
    In other words, clusters should be general-purpose and even. Thus, we use Bisecting K-Means Clustering.

    Then output to the `out_file`.

    :param in_file: file input
    :param out_file: file output
    :param cuda_devices: specify visible GPU device. If you have multiple GPUs, you can specify some of them (e.g. 0, 1, 2) or all of them (e.g., all).
    :param kwargs:other parameters including,
        1) emb_model: model used to get embeddings for definitions of type words
        2) emb_bs: batch size for sent_emb_model.
        3) local_files_only
        4) n_clusters: cluster number for clustering.
        5) init: Method for initialization, we can choose from {'k-means++', 'random'} or a callable.
        6) n_init: Number of time the inner k-means algorithm will be run with different centroid seeds in each bisection.
        7) random_state: the random seed, which use an int to make the randomness deterministic
        8) bisecting_strategy: Defines how bisection should be performed; choose from ("biggest_inertia", "largest_cluster")

    :return:
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
    # set GPU device
    if cuda_devices == 'all':
        # set the GPU can be used
        device_ids = [i for i in range(torch.cuda.device_count())]
        cuda_devices = [str(i) for i in range(torch.cuda.device_count())]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        device_ids = [int(i) for i in ','.split(cuda_devices)]
    device = torch.device("cuda:0")  # set the main GPU device

    # 1. Import embedding models. It will download the models automatically
    emb_tokenizer = AutoTokenizer.from_pretrained(kwargs['emb_model'], local_files_only=kwargs['local_files_only'])
    emb_model = AutoModel.from_pretrained(kwargs['emb_model'], local_files_only=kwargs['local_files_only'])
    if torch.cuda.device_count() > 1:  # use multi-GPU
        emb_model = torch.nn.DataParallel(emb_model, device_ids=device_ids)
    emb_model.eval()
    emb_model.to(device)

    # 2. Preprocess and tokenize the definitions of the input type words
    # 2.1. get type words' definitions
    all_definitions, all_definitions_embbedings = [], []  # store all definition and its embeddings
    word_defi_id_pair = dict()  #  Some type word have more than one definition. We use word_defi_id_pair to track type word and its definition
    defi_id = 0  # index all definition.
    input_type_info = dict()  # store the input type information
    with jsonlines.open(in_file, mode='r') as in_file:
        for line in in_file:
            word_id, definitions = str(line['id']), line['definitions']
            del line['id']
            input_type_info.update({word_id: line})  # e.g. {1:{"word": "body_part", "source": "other", "definitions": ["a part of a human body."]}}
            for defi in definitions:
                if defi != 'None':
                    all_definitions.append(defi)
                    word_defi_id_pair.update({defi_id: word_id})  # e.g. {0:0, 1:0} means that the word (0) has two definition (defi 0 and defi 1)
                    defi_id += 1

    # 2.2. get embeddings of the whole definitions
    defi_loader = DataLoader(all_definitions, batch_size=kwargs['emb_bs'], shuffle=False)
    with torch.no_grad():
        for batch_defs in defi_loader:
            batch_inputs = emb_tokenizer(batch_defs, padding=True, truncation=True, return_tensors="pt")
            batch_inputs.to(device)
            all_definitions_embbedings += emb_model(**batch_inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

    # 3. Bisecting K-Means Clustering
    # 3.1 We use Bisecting K-Means Clustering to get general-purpose and even cluster for the whole definitions
    # see detail at https://scikit-learn.org/stable/modules/clustering.html#bisecting-k-means
    # and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn-cluster-bisectingkmeans
    cluster_labels =  BisectingKMeans(n_clusters=kwargs['n_clusters'],
                                      init=kwargs['init'],
                                      n_init=kwargs['n_init'],
                                      random_state=kwargs['random_state'],
                                      bisecting_strategy=kwargs['bisecting_strategy']).fit_predict(all_definitions_embbedings)

    # 3.2 collect definition and it's id according clustering
    clusters = [[] for _ in range(kwargs['n_clusters'])]
    for defi_id, (defi, cluster_label) in enumerate(zip(all_definitions, cluster_labels)):
        clusters[cluster_label].append({'defi_id': defi_id, 'definition': defi})

    # 3.3 clear unused GPU memory
    del emb_model
    torch.cuda.empty_cache()

   # 4. judge by LLMs. We use vLLM for faster inference
    redundant_id = judge_by_llm(clusters, input_type_info, **kwargs)

    # 5 filter and output the result
    # 5.1 filter the redundant definition according the redundant_id
    # e.g. [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 3)] -> [(0, 0), (2, 0), (3, 1), (6, 3)]
    result = filter(lambda defi_id, _ : defi_id not in redundant_id, word_defi_id_pair.items())
    with jsonlines.open(out_file, 'w') as writer:

        # group by the original word_id (the second element of the tuple in the word_defi_id_pair)
        # e.g. [(0, 0), (2, 0), (3, 1), (6, 3)] -> [(0, 0), (2, 0)] | [(3, 1)] | [(6, 3)]
        # key is the original word_id like 0, group is the definition ids of the word_id like [(0, 0), (2, 0)]
        for new_word_id, (orig_word_id, group) in enumerate(itertools.groupby(result, key=lambda x: x[1])):
            definitions = [all_definitions[defi_id] for defi_id, _ in group]
            writer.write({'id': new_word_id,
                          'word': input_type_info[orig_word_id]['word'],
                          'source': input_type_info[orig_word_id]['source'],
                          'definitions': definitions})


if __name__ == '__main__':
    # test code
    query = 'living_thing'
    api_url = 'https://api.dictionaryapi.dev/api/v2/entries/'
    lang_type = 'en'
    sentences = 2
    print(get_type_word(query,
                        api_url=api_url,
                        lang_type=lang_type,
                        sentences=sentences,
                        auto_suggest=False))