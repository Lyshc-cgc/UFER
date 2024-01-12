"""
This module is used to:
1) Make request to a free dictionary api {https://github.com/meetDeveloper/freeDictionaryAPI}, which is backend
    with Wiktionary {https://en.wiktionary.org/wiki/Wiktionary:Main_Page}.
2) Parse the response we need into specific object. We use noun meaning of a word as the definition of the word.
3) If cannot find the query word at the free dictionary api, we will try to search on Wikipedia.
4) We use the first two sentences of Wikipedia Encyclopedia corresponding to that word as the definition.
"""

import requests
import json
import wikipedia as wp
from dataclasses import dataclass
from typing import List, Optional


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

if __name__ == '__main__':
    # test code
    query = 'living_thing'
    api_url = 'https://api.dictionaryapi.dev/api/v2/entries/'
    lang_type = 'en'
    sentences = 2
    print(get_type_word(query, api_url=api_url, lang_type=lang_type, sentences=sentences))