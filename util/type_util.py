"""
This module is used to help build the type information of the UFER dataset, including:
1) get definition for type words in the UFER dataset.
    a) Make request to a free dictionary api {https://github.com/meetDeveloper/freeDictionaryAPI}, which is backend
    with Wiktionary {https://en.wiktionary.org/wiki/Wiktionary:Main_Page}.
    b) Parse the response we need into specific object. We use noun meaning of a word as the definition of the word.
    c) If we cannot find the query word at the free dictionary api, we will try to search on Wikipedia.
    d) We use the first two sentences of Wikipedia Encyclopedia corresponding to that word as the definition.
2) remove unsuitable type words by disambiguating.
"""
import ast
import copy
import re
import os
import requests
import json
import jsonlines
import wikipedia as wp
import pandas as pd
import torch
import itertools
import numpy as np
from collections import Counter
from vllm import LLM, SamplingParams
from sklearn.cluster import BisectingKMeans
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
from util.func_util import get_device_ids, batched

@dataclass
class TypeWord:
    """
    Class TypeWord used to store the information (id and definition) of each type word in the UFER dataset.
    """
    id: int  # id of the type word
    word: str  # the type word
    source: str  # The source of the definitions, should be one of ('dic_api', 'wiki', 'other').
    definitions: List[str]  # definitions of the type word

@dataclass
class TypeWordv2:
    """
    Class TypeWordv2 used to store the information (id and definition) of each type word in the UFER dataset.
    The definition of a type word is the most frequent definition of the word.
    """
    id: int  # id of the type word
    word: str  # the type word
    source: str  # The source of the definitions, should be one of ('dic_api', 'wiki', 'other').
    definition: str  # definitions of the type word

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
    if os.path.exists(out_file):
        print(f"{out_file} exists, we don't need to delete the type words with the definition 'None' again.")
        return
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

def get_defi_by_llm(in_file, device_ids, **def_model_cfg):
    """
    get the best definition of each type word by LLMs.
    :param in_file:
    :param device_ids: The ids of GPUs you want use to inference
    :param def_model_cfg: the configuration of the def model used to get the best definition.
    :return:
    """
    def _generate_chat_messages(in_file, chat_msg_template, **kwargs):
        """
        Generate chat messages for each type word in the input file.
        :param in_file: input file containing the type words with multiple definitions
        :param chat_msg_template: the chat message template for the model to generate the chat message.
        :param kwargs:
        :return:
        """
        with jsonlines.open(in_file) as reader:
            for line in reader:
                type_word = TypeWord(**line)
                if len(type_word.definitions) <= 1:  # skip the type words with only one definition
                    continue
                chat_msg = copy.deepcopy(chat_msg_template)
                defis = []
                for idx, defi in enumerate(type_word.definitions):
                    defi = kwargs['def_template'].format(id=idx, definition=defi)
                    defis.append(defi)
                usr_content = kwargs['usr_prompt'].format(word=type_word.word, definitions=' '.join(defis))
                chat_msg.append({"role": "user", "content": usr_content})
                yield chat_msg

    # 1. init chat message template
    if def_model_cfg['name'] == 'Qwen':
        chat_msg_template = [
            {"role": "system", "content": def_model_cfg['sys_prompt']},
        ]
        for example in def_model_cfg['examples']:
            usr_content = def_model_cfg['usr_prompt'].format(word=example['word'], definitions=example['definitions'])
            chat_msg_template.append({"role": "user", "content": usr_content})
            chat_msg_template.append({"role": "assistant", "content": str(example['output'])})
    else:
        # e.g., for Mistral and Yi, we only need to input the user's prompt
        chat_msg_template = []

    # 2. init llm and its tokenizer
    def_model = LLM(model=def_model_cfg['checkpoint'],
                    tensor_parallel_size=len(device_ids),
                    dtype=def_model_cfg['dtype'],
                    gpu_memory_utilization=def_model_cfg['gpu_memory_utilization'])
    # get its tokenizer to apply the chat template
    # https://github.com/vllm-project/vllm/issues/3119
    tokenizer = def_model.llm_engine.tokenizer.tokenizer
    sampling_params = SamplingParams(temperature=def_model_cfg['temperature'], top_p=def_model_cfg['top_p'], max_tokens=def_model_cfg['max_tokens'])

    # 3. main process
    assert in_file.endswith('.jsonl'), f"Input file should be a jsonl file, but got {in_file}"
    pbar = tqdm(batched(_generate_chat_messages(in_file, chat_msg_template, **def_model_cfg),
                        def_model_cfg['bs']),
                desc=f"get best definition by {def_model_cfg['name']}", )
    json_pattern = r'\{\{(.*?)\}\}'  # the pattern to extract JSON string
    each_results = []  # store the judge result by each llm
    for batch_chats in pbar:
        # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
        templated_batch_chats = tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
        outputs = def_model.generate(templated_batch_chats, sampling_params)
        for out_id, output in enumerate(outputs):
            # only extract the first JSON string
            result = re.findall(json_pattern, output.outputs[0].text, re.DOTALL)
            if result and len(result) >= 1:
                try:
                    each_results.append(json.loads('{' + result[0] + '}'))
                except json.JSONDecodeError:
                    print('{' + result[0] + '}')
                    each_results.append({"analysis": None, "answer": None})
            else:
                print(output.outputs[0].text)
                each_results.append({"analysis": None, "answer": None})
    return each_results

def get_best_definition(in_file, out_file, cuda_devices, **kwargs):
    """
    read the `in_file` and get the best definition of each type word, output to the `out_file`.
    :param in_file: input file containing the type words with multiple definitions
    :param out_file: output file containing the type words with the best definition
    :param kwargs:
        1) def_models: configurations of the def models used to get the best definition.
        2) work_dir: the work directory.
    :return:
    """
    if os.path.exists(out_file):
        print(f"{out_file} exists, we don't need to get the best definition again.")
        return # if the output file exists, we don't need to get the best definition again

    # 0. Some settings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
    device_ids = get_device_ids(cuda_devices)
    results = []  # store the judge results from all def models

    # 1. get best definition by each LLM
    for def_model_cfg in kwargs['def_models']:
        # Attention, this for loop is used to get the best definition by each def model
        # However, because some unknown reason about vLLm or Ray, we cannot empty the GPU memory after each loop
        # So, we should manually run this for loop for each def model and cache their results.

        # 1.1. check the cache file
        model_work_dir = os.path.join(kwargs['work_dir'], def_model_cfg['name'])
        cache_file_name = def_model_cfg['def_res_cache_file']
        def_res_cache_file = os.path.join(model_work_dir, cache_file_name)  # the file to cache the results from all def models

        if def_res_cache_file.endswith('.jsonl') and os.path.exists(def_res_cache_file):  # read the cache file directly
            with jsonlines.open(def_res_cache_file, 'r') as reader:
                cache_res = [line for line in reader]
            results.append(cache_res)
            continue

        # else we should get the best definition by this LLM from scratch
        # 1.2 get result by this LLM
        each_results = get_defi_by_llm(in_file, device_ids, **def_model_cfg)
        results.append(each_results)

        # 1.3 cache the result by each llm
        if not os.path.exists(model_work_dir):
            os.makedirs(model_work_dir)
        with jsonlines.open(def_res_cache_file, 'w') as writer:
            for res in each_results:
                writer.write(res)

    # 2. output
    with jsonlines.open(in_file, 'r') as reader, jsonlines.open(out_file, 'w') as writer:
        idx = 0  # index for those type words with multiple definitions (> 2)
        for line in reader:
            if len(line['definitions']) <= 1:
                # For those type words with only two definition
                # copy the first definition directly
                line['definition'] = line['definitions'][0]
            else:
                if not results[0][idx]:
                    # No correct answer generated by LLMs
                    # copy the first definition directly
                    line['definition'] = line['definitions'][0]
                else:  # For those type words with multiple definitions, we should get the best definition by voting
                    line['definition'] = results[0][idx]['answer']
                idx += 1
            del line['definitions']
            writer.write(line)


def judge_by_llm(work_dir, clusters, device_ids, **kwargs):
    """
    Judge the similarity of the definitions in each cluster by LLMs and remove the redundant definitions from each cluster.

    :param work_dir: The work directory for this judge model.
    :param clusters: The clusters of the definitions of the type words. Each cluster is a list of dict, where each dict
        contains the id of the definition and the definition.
    :param device_ids: The ids of GPUs you want use to inference
    :param kwargs:
        0) name: the name of the LLM used to judge.
        1) checkpoint: the LLM checkpoint used to judge. model_name or path.
        2) sys_prompt: (Optional) prompt for the judge model. It is used as the 'system' role of the chat message.
        3) examples: (Optional) examples for the judge model to generate the chat message. It is used as the 'user' and 'assistant' role of the chat message.
        4) usr_prompt: prompt for the judge model. It should be the 'user' role of the chat message.
        5) verify_prompt: prompt for the judge model to verify the answer.  We need to continue the conversation to
            ask for the correct answer
        6) temperature: temperature for the judge model. Default is 0.
        7) top_p: top_p for the judge model. he smaller the value, the more deterministic the model output is.
        8) max_tokens: The maximum number of tokens to generate.
        9) bs: batch size for the LLM.
        10) dtype: The data type of the input tensor. https://docs.vllm.ai/en/latest/models/engine_args.html#cmdoption-dtype
        11) jud_res_cache_file: the file to cache the judge results.
        12) ver_res_cache_file: the file to cache the judge results after verification.
        13) verify_convs_cache_file: the file to cache the verification conversations.
    :return: List[int], The redundant definitions' ids.
    """
    jud_model_name = kwargs['name']
    print("="*20 + f"Start to judge by LLMs: {jud_model_name}" + "="*20)
    out_dir = os.path.join(work_dir, jud_model_name)  # the output directory of this judge model
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ver_res_cache_file = os.path.join(out_dir, kwargs['ver_res_cache_file'])
    if not os.path.exists(ver_res_cache_file):  # if judge result (after verification) doesn't exist, we should judge from scratch
        # 1. judge by LLMs. We use vLLM for faster inference
        # 1.1 Import the judge model
        # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
        jud_model = LLM(model=kwargs['checkpoint'],
                        tensor_parallel_size=len(device_ids),
                        dtype=kwargs['dtype'],
                        )
        # get its tokenizer to apply the chat template
        # https://github.com/vllm-project/vllm/issues/3119
        jud_tokenizer = jud_model.llm_engine.tokenizer.tokenizer
        sampling_params = SamplingParams(temperature=kwargs['temperature'], top_p=kwargs['top_p'],
                                         max_tokens=kwargs['max_tokens'])

        # 1.2 Remove definitions with duplicate meanings in each cluster by pairwise comparison
        # 1.2.1 prepare the chats input to the judge model
        chats = []  # store the chats
        defi_As, defi_Bs = [], []  # store the first and second def of each pair respectively
        pair_num_of_clusters = []  # store the number of pairs of each cluster

        # 1.2.2 prepare the chat message in different format according to the judge model
        if jud_model_name == 'Qwen':
            chat_msg_template = [
                {"role": "system", "content": kwargs['sys_prompt']},
            ]
            for example in kwargs['examples']:
                usr_content = kwargs['usr_prompt'].format(first_definition=example['definition_A'],
                                                          second_definition=example['definition_B'])
                chat_msg_template.append({"role": "user", "content": usr_content})
                chat_msg_template.append({"role": "assistant", "content": str(example['output'])})
        else:
            # e.g., for mistral and Yi, we only need to input the user's prompt
            chat_msg_template = []

        for cluster in clusters:
            # get definition pair in this clustering
            # https://docs.python.org/3.10/library/itertools.html#itertools.combinations
            defi_pair = list(itertools.combinations(cluster, 2))
            pair_num_of_clusters.append(len(defi_pair))
            for defi_A, defi_B in defi_pair:  # process the chats
                defi_As.append(defi_A)
                defi_Bs.append(defi_B)

                tmp_chat_msg = copy.deepcopy(chat_msg_template)
                usr_content = kwargs['usr_prompt'].format(first_definition=defi_A['definition'], second_definition=defi_B['definition'])
                tmp_chat_msg.append({"role": "user", "content": usr_content})
                chats.append(tmp_chat_msg)

        # 1.2.2 prepare the batched chats input to the judge model, and then judge
        jud_results = []  # store the judge result
        verify_convs = []  # store the verification conversations need to be continued to ask for the correct answer
        verify_conv_ids = []  # store the id of the verification conversations need to be continued
        pattern = r'\d+'  # pattern to match the number in the output

        jud_res_cache_file = os.path.join(out_dir, kwargs['jud_res_cache_file'])  # the file to cache the judge results
        verify_convs_cache_file = os.path.join(out_dir, kwargs['verify_convs_cache_file'])  # the file to cache the verification conversations
        if not os.path.exists(jud_res_cache_file) or not os.path.exists(verify_convs_cache_file):  # doesn't exist, we should judge from scratch
            with tqdm(total=len(chats), desc='judging') as t:
                for batch_id, batch_chats in enumerate(batched(chats, kwargs['bs'])):
                    # we should use tokenizer.apply_chat_template to add generation template to the chats explicitly
                    templated_batch_chats = jud_tokenizer.apply_chat_template(batch_chats, add_generation_prompt=True, tokenize=False)
                    outputs = jud_model.generate(templated_batch_chats, sampling_params)  # judge
                    for out_id, output in enumerate(outputs):
                        conv_id = batch_id * kwargs['bs'] + out_id  # the id of the conversation
                        out_answer = output.outputs[0].text
                        if jud_result := re.search(pattern, out_answer):  # there is a number in the output
                            jud_result = int(jud_result.group())
                            if jud_result in (1, 0):  # LLM generated correct result
                                jud_results.append(jud_result)  # here, jud_result is 1 or 0
                                continue
                        # There are 2 cases to execute the following code:
                        # 1) There is no number in the output,
                        # 2) The number is not 1 or 0, which means that the LLM generated wrong result
                        #  We should store those verification conversations, which need to be continued to ask for the correct answer.
                        jud_results.append(-1)  # label the result as -1, which means that the LLM didn't generated answer we needed

                        # batch_chats[out_id] is a chat message history need to be continued, like [{"role": "user", "content": <prompt>}]
                        # we need to append the new message to it
                        batch_chats[out_id].append({"role": "assistant", "content": out_answer})  # append the LLM's answer
                        batch_chats[out_id].append({"role": "user", "content": kwargs['verify_prompt']})  # append the verification prompt
                        verify_convs.append(jud_tokenizer.apply_chat_template(batch_chats[out_id], tokenize=False, add_generation_template=True))
                        verify_conv_ids.append(conv_id)
                    t.update(kwargs['bs'])

            # 1.2.3 cache the judge results and the verification conversations
            cache_jud_results = {'jud_results': jud_results, 'definition_A': defi_As, 'definition_B': defi_Bs}
            cache_verify_convs = {'verify_conv_ids': verify_conv_ids, 'verify_convs': verify_convs}
            pd.DataFrame(cache_jud_results).to_csv(jud_res_cache_file, index_label='id')
            pd.DataFrame(cache_verify_convs).to_csv(verify_convs_cache_file, index_label='id')
        else:  # read the judge result from the cache file directly
            cache_jud_results = pd.read_csv(jud_res_cache_file)
            jud_results = cache_jud_results['jud_results'].tolist()

            cache_verify_convs = pd.read_csv(verify_convs_cache_file)
            verify_conv_ids = cache_verify_convs['verify_conv_ids'].tolist()
            verify_convs = cache_verify_convs['verify_convs'].tolist()

        # 1.2.4 start to verify We need to continue the conversation to ask for the correct answer
        with tqdm(total=len(verify_convs), desc='verifying') as t:
            for batch_convs, batch_conv_ids in zip(batched(verify_convs, kwargs['bs']),
                                                   batched(verify_conv_ids, kwargs['bs'])):
                # the batch_cons are applied templated in Line 275, so we don't need to apply template again
                outputs = jud_model.generate(batch_convs, sampling_params)  # verify
                for conv_id, output in zip(batch_conv_ids, outputs):
                    out_answer = output.outputs[0].text
                    if jud_result := re.search(pattern, out_answer):  # there is a number in the output
                        jud_result = int(jud_result.group())
                        if jud_result in (1, 0):  # LLM generated correct result
                            jud_results[conv_id] = jud_result  # here, jud_result is 1 or 0
                            continue
                    raise ValueError(f"LLM didn't generate answer we needed: {jud_result}")
                t.update(kwargs['bs'])

        # 1.2.5 cache the results after verification
        cache_jud_results = {'jud_results': jud_results, 'definition_A': defi_As, 'definition_B': defi_Bs}
        pd.DataFrame(cache_jud_results).to_csv(ver_res_cache_file, index_label='id')
        del jud_model
        torch.cuda.empty_cache()
    else:  # read the judge result (after verification) from the cache file directly
        print(f"Read directly the judge result (after verification) from the cache file: {ver_res_cache_file}")
        cache_ver_jud_results = pd.read_csv(ver_res_cache_file)
        jud_results = cache_ver_jud_results['jud_results'].tolist()
        defi_As, defi_Bs = cache_ver_jud_results['definition_A'].tolist(), cache_ver_jud_results['definition_B'].tolist()
        defi_As, defi_Bs = map(ast.literal_eval, defi_As), map(ast.literal_eval, defi_Bs)  # convert string to dict

    # 1.2.6 Remove similar defs according to the judge results
    definition_pairs = zip(defi_As, defi_Bs)
    redundant_id = []  # to store the id of the redundant definition

    # only leave pairs with similar definitions
    # https://docs.python.org/3.10/library/itertools.html#itertools.compress
    # e.g. [(A,B), (A,C), (A,D), (A,E), (B,C), (B,D), (B,E), (C,D), (C,E)] -> [(A, B), (A, D), (B, D), (C, E)]
    # It means that obj A, B, and D have similar meanings, which is the same applies to obj C and E.
    # The judge result is a list like [1,0,1,0,0,1,0,0,1], where 1 means that the definition are similar in that pair.
    similar_pairs = itertools.compress(definition_pairs, jud_results)

    # Store the id of the redundant definition
    # group by the ID (i.e, defi_id) of the first definition in a pair
    # https://docs.python.org/3.10/library/itertools.html#itertools.groupby
    # e.g. [(A, B), (A, D), (B, D), (C, E)] -> (A, B), (A, D) | (B, D) | (C, E)
    for key, group in itertools.groupby(similar_pairs, key=lambda x: x[0]['defi_id']):
        # e.g. (A, B), (A, D) -> (A, B, D)
        simi_definition_id_set = set()  # A set to store the ids of similar definitions
        for pair in group:
            for e in pair:
                simi_definition_id_set.add(e['defi_id'])
        tmp_redundant_id = sorted(list(simi_definition_id_set))
        # We leave first definition in each group, and consider others as the redundant definitions to be removed
        redundant_id += tmp_redundant_id[1:]

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
    :param kwargs: The type cfg, we will use the following parameters in this method:
        1) emb_model: model used to get embeddings for definitions of type words
        2) emb_bs: batch size for sent_emb_model.
        3) local_files_only
        4) n_clusters: cluster number for clustering.
        5) init: Method for initialization, we can choose from {'k-means++', 'random'} or a callable.
        6) n_init: Number of time the inner k-means algorithm will be run with different centroid seeds in each bisection.
        7) random_state: the random seed, which use an int to make the randomness deterministic
        8) bisecting_strategy: Defines how bisection should be performed; choose from ("biggest_inertia", "largest_cluster")
        9) jud_models: A list of the judge models we used.

    :return:
    """
    if os.path.exists(out_file):
        print(f"{out_file} exists, we don't need to disambiguate type words again.")
        return
    # 0. Some settings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
    # set GPU device
    device_ids = get_device_ids(cuda_devices)
    main_device = str(device_ids[0])
    device = torch.device(f"cuda:{main_device}")  # set the main GPU device

    # 1. Preprocess the definition of the input type words
    # 2.1 get type words' definition
    all_definitions = []  # store all definitions
    type_info = []  # store the information of the type words
    with jsonlines.open(in_file, mode='r') as in_file:
        for line in in_file:
            type_info.append(line)
            all_definitions.append(line['definition'])

    # 2. Embedding and Clustering
    # 2.1 read cached cluster results or do embedding and clustering from scratch
    emb_and_cluster_flag = False  # whether to do embedding and clustering
    cluster_res_cache_file = os.path.join(kwargs['work_dir'], kwargs['cluster_res_cache_file'])
    if os.path.exists(cluster_res_cache_file):
        cluster_results = pd.read_csv(cluster_res_cache_file)
        cluster_labels = cluster_results['cluster_label'].tolist()
        if len(set(cluster_labels)) != kwargs['n_clusters']:
        # check the cluster number.
        # If it's not equal to n_clusters, it means that the cached cluster results are not suitable for the current n_clusters.
            emb_and_cluster_flag = True
    else:
        emb_and_cluster_flag = True

    # if there is cached cluster results and the cluster number is suitable for the current n_clusters,
    # we don't need to do embedding and clustering
    if emb_and_cluster_flag:
        # 2.2 Embedding
        # 2.2.1 Import embedding models.
        emb_tokenizer = AutoTokenizer.from_pretrained(kwargs['emb_model'], local_files_only=kwargs['local_files_only'])
        emb_model = AutoModel.from_pretrained(kwargs['emb_model'], local_files_only=kwargs['local_files_only'])
        if torch.cuda.device_count() > 1:  # use multi-GPU
            emb_model = torch.nn.DataParallel(emb_model, device_ids=device_ids)
        emb_model.eval()
        emb_model.to(device)

        # If there is no cached cluster results, we should get it from scratch
        # 2.2.2 get embeddings of the whole definitions
        all_definitions_embbedings = []  # store the embeddings of the whole definitions
        with torch.no_grad():
            for batch_defs in batched(all_definitions, kwargs['emb_bs']):
                batch_inputs = emb_tokenizer(batch_defs, padding=True, truncation=True, return_tensors="pt")
                batch_inputs.to(device)
                all_definitions_embbedings += emb_model(**batch_inputs, output_hidden_states=True, return_dict=True).pooler_output.cpu()

        # 2.3 Bisecting K-Means Clustering
        # 2.3.1 We use Bisecting K-Means Clustering to get general-purpose and even cluster for the whole definitions
        # see detail at https://scikit-learn.org/stable/modules/clustering.html#bisecting-k-means
        # and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn-cluster-bisectingkmeans
        cluster_labels =  BisectingKMeans(n_clusters=kwargs['n_clusters'],
                                          init=kwargs['init'],
                                          n_init=kwargs['n_init'],
                                          random_state=kwargs['random_state'],
                                          bisecting_strategy=kwargs['bisecting_strategy']).fit_predict(all_definitions_embbedings)
        cluster_results = {'definition': all_definitions, 'cluster_label': cluster_labels}
        pd.DataFrame(cluster_results).to_csv(cluster_res_cache_file, index_label='id')

        # 2.3.2 clear unused GPU memory
        del emb_model
        torch.cuda.empty_cache()

    # 2.4 collect definition and it's id according clustering
    clusters = [[] for _ in range(kwargs['n_clusters'])]
    for defi_id, (defi, cluster_label) in enumerate(zip(all_definitions, cluster_labels)):
        clusters[cluster_label].append({'defi_id': defi_id, 'definition': defi})

   # 3. judge by LLMs. We use vLLM for faster inference
    redundant_id = []
    for jud_model_cfg in kwargs['jud_models']:
        # kwargs['work_dir'] is the work directory
        # kwargs[jud_model] is the config of the judge model
        redt_id_from_this_model = judge_by_llm(kwargs['work_dir'], clusters,device_ids, **jud_model_cfg)
        redundant_id += redt_id_from_this_model

    # 4 filter and output the result
    # 4.1 Count the number of votes for each redundant id, we only keep the redundant id which has more than half votes
    # e.g. [0, 0, 1, 1, 2, 3, 3] -> {0: 2, 1: 2, 2: 1, 3: 2}, key is the redundant id, value is the number of votes
    # https://docs.python.org/3.10/library/collections.html?highlight=counter#counter-objects
    counter = Counter(redundant_id)
    # voting_res = filter(lambda x: x[1] >= len(kwargs['jud_models']) / 2, counter.items())
    voting_res = filter(lambda x: x[1] > 0, counter.items())
    redundant_id = [x[0] for x in voting_res]  # key is the redundant id

    # 4.2 filter the redundant definition according the redundant_id by voting from different judge models
    with jsonlines.open(out_file, 'w') as writer:
        new_word_id = 0
        for line in type_info:
            if line['id'] in redundant_id:  # skip redundant type words
                continue
            writer.write({'id': new_word_id,
                          'word': line['word'],
                          'source': line['source'],
                          'definition': line['definition']})

    print(f"store the disambiguated type words in the file: {out_file}")

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