# -*- coding: utf-8 -*-
# pylint: disable=unused-variable
import csv
import re
import string
import sys
import unicodedata
from functools import reduce
import random
import os

import cmudict
import emoji
import emot
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, TweetTokenizer
from spellchecker import SpellChecker
import arff
import tempfile
import json

from sklearn import preprocessing

class CANLTK:
    stops = stopwords.words('english')
    arpabet = cmudict.dict()
    twitter_tokenizer = TweetTokenizer()

    @staticmethod
    def n_lower_chars(string):
        return sum(map(str.islower, string))

    @staticmethod
    def n_upper_chars(string):
        return sum(map(str.isupper, string))

    @staticmethod
    def n_isspace_chars(string):
        return sum(map(str.isspace, string))

    @staticmethod
    def n_vowels_chars(string):
        return sum(map(string.lower().count, "aeiou"))

    @staticmethod
    def n_count_and_print_alphabets(CONFIG, string):
        CONFIG['feature_count']
        d = {}
        uppercaseAlphabetArray = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ]
        lowercaseAlphabetArray = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]
        #     print(len(uppercaseAlphabetArray))
        #     print(len(lowercaseAlphabetArray))
        for i in range(26):
            combo = uppercaseAlphabetArray[i] + lowercaseAlphabetArray[i]
            CONFIG['feature_count']+=1
            d["f" + (str(CONFIG['feature_count'])) + "-" + combo] = len(re.findall('[' + combo + ']' , string))
        return d

    @staticmethod
    def n_special_chars(string):
        return sum(map(string.lower().count, ".,?!<>@#$%&()[]:;\'\""))

    @staticmethod
    def n_count_and_print_special_chars(CONFIG, string):
        CONFIG['feature_count']
        d = {}
        i = 0
        specialChars = [".", ",", "—", "–", "’", "‘", "?", "!", "<", ">", "@", "#", "$", "%", "&", "(", ")", "[", "]", ":", ";", "\'", "\""]
        specialCharsNamed = ["fullstop", "comma", "em-dash", "en-dash", "right-single-quotation-mark", "left-single-quotation-mark", "question-mark", "exclamation", "less-than-sign", "greate-than-sign", "at-sign", "hash", "dollar", "percentage", "ampersand", "open-brackets", "closing-brackets", "open-sq-brackets", "close-sq-brackets", "colan", "semi-colan", "single-quotes", "double-quotes"]
        for c in specialChars:
            CONFIG['feature_count']+=1
            d["f" + str(CONFIG['feature_count']) + "-p-" + specialCharsNamed[i]] =len(re.findall('[' + c + ']' , string))
            i+=1
        return d

    @staticmethod
    def extract_emojis(str):
        return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

    @staticmethod
    def n_long_words(wordTokens):
        filteredTokens = list(filter(lambda x: len(x) > 6, wordTokens))
        return len(filteredTokens)

    @staticmethod
    def n_words_le_3(wordTokens):
        filteredTokens = list(filter(lambda x: len(x) <= 3, wordTokens))
        return len(filteredTokens)

    @staticmethod
    def n_words_le_2(wordTokens):
        filteredTokens = list(filter(lambda x: len(x) <= 2, wordTokens))
        return len(filteredTokens)

    @staticmethod
    def avg_words(wordTokens):
        count = 0
        for token in wordTokens:
            count += len(token)
        if len(wordTokens) != 0:
            return count / len(wordTokens)
        else:
            return 0
        
    @staticmethod
    def n_lowercase_sentences(sentTokens):
        count = 0
        for token in sentTokens:
            if token[0].islower():
                count += 1
        return count
        
    @staticmethod
    def n_uppercase_sentences(sentTokens):
        count = 0
        for token in sentTokens:
            if token[0].isupper():
                count += 1
        return count

    @staticmethod
    def n_each_emoticons(string):
        d = {}
        # pylint: disable=unused-variable
        for key, value in emot.EMOTICONS.items():
            d[key] = string.count(key)
        return d

    @staticmethod
    def n_each_emojis(string):
        d = {}
        # pylint: disable=unused-variable
        for key, value in emoji.UNICODE_EMOJI.items():
            d[key] = string.count(key)
        return d
     
    @staticmethod       
    def n_misspelled_words(sentTokens):
        spell = SpellChecker()
        # single proper punctuation isn't regarderded as mispelled multiple false punctuation is considered as mispelled
        #     misspelled = spell.unknown(['somessthing', 'is', 'hapenning', 'here', '!', '👍', ',', ':)'])
        misspelled = spell.unknown(sentTokens)
        return len(misspelled)

    @staticmethod
    def n_total_punctuations(text):
        count = 0
        for p in string.punctuation:
            count += text.count(p)
        return count

    @staticmethod
    def print_n_each_punctuation(CONFIG, text):
        CONFIG['feature_count']
        # pylint: disable=unused-variable
        d = {}
        count = 0
        i = 0
        specialCharsNamed = ["exclamation", "double-quotes", "hash", "dollar", "percentage", "ampersand", "single-quotes", "open-brackets", "closing-brackets", "asterix", "plus", "comma", "dash", "fullstop", "slash", "colan", "semi-colan", "less-than-sign", "equal-sign", "greater-than-sign", "question-mark", "at-sign", "open-sq-brackets", "back-slash", "close-sq-brackets", "caret", "underscore", "grave-accent", "open-curly-brace", "vertical-bar", "close-curly-brace", "tilde"]
        for p in string.punctuation:
            CONFIG['feature_count']+=1
            d["f" + str(CONFIG['feature_count']) + "-p-" + specialCharsNamed[i]] = text.count(p)
            i+=1
        return d

    @staticmethod
    def avg_syllables(sentTokens):
        #     print(sentTokens)
        # emoticons like O.o scews the data up, also Okay👍👍🏿 as a single word so doesnt count syllables there need to cleanse and send
        validWords = 0
        count = 0
        for token in sentTokens:
            token = token.lower()
            if token in CANLTK.arpabet:
        #             print(token)
        #             print(arpabet[token])
                count += len(CANLTK.arpabet[token][0])
                validWords += 1
        if validWords != 0:
            return count/validWords
        else:
            return 0
    # https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules
      
    @staticmethod      
    def n_punctuation(string):
        # need to cleanse emojis
        count = 0
        for c in string:
            if "P" in unicodedata.category(c):
        #             print(c, unicodedata.category(c))
                count += 1
        return count

    @staticmethod
    def prune_emojis_emoticons(string):
        # at tim doesn't work specially when emoticons comes after a weird emoji like O.o which isnt registered
        if "location" in emot.emoji(string).keys() is not None:
            for loc in reversed(emot.emoji(string)['location']):
                string = string[0:loc[0]] + string[loc[1] + 1::]
        #     print(emot.emoticons(string))
        if "location" in emot.emoticons(string):
            for loc in reversed(emot.emoticons(string)['location']):
                string = string[0:loc[0]] + string[loc[1] + 1::]
        return string

    @staticmethod
    def n_function_words(string):
        count = 0
        for f in string:
            if f in CANLTK.stops:
                count += 1
        return count
      
    @staticmethod          
    def n_context_words(string):
        count = 0
        for f in string:
            if f not in CANLTK.stops:
                count += 1
        return count

    # def count__each_most_common_words(string):
    #     d = {}
    #     freq_dist = FreqDist(wordTokens)
    #     for li in freq_dist.most_common(10):
    #         li

    @staticmethod
    def n_total_emoticons(string):
        if any("value" in d for d in emot.emoticons(string)):
            return len(emot.emoticons(string)["value"])
        else:
            return 0

    @staticmethod
    def n_total_emojis(string):
        if any("value" in d for d in emot.emoji(string)):
            return len(emot.emoji(string)["value"])
        else:
            return 0

    @staticmethod
    def prune_punctuations_special_characters(text_arr):
        # need to recorrect
        for text in text_arr:
            if text in string.punctuation:
                text_arr.remove(text)
        return text_arr

    @staticmethod
    def prune_function_words(string_arr):
        items_to_remove = []
        for string in string_arr:
            string_lowercase = string.lower()
            if string_lowercase in CANLTK.stops:
                items_to_remove.append(string)
        for rm in items_to_remove:
            string_arr.remove(rm)
        return string_arr

    @staticmethod
    def n_words(string, string_arr):
        count = 0
        # for f in string:
        if string in string_arr:
            count += 1
        return count

    @staticmethod
    def replace_keys(string):
        return string.replace("!", "exclamation").replace("\"", "double-quotes").replace("$", "dollar").replace("%", "percentage").replace("&", "ampersand").replace("'", "single-quotes").replace("(","open-brackets").replace(")", "closing-brackets").replace("*", "asterix").replace("+","plus").replace(",","comma").replace("-","dash").replace(".","fullstop").replace("/","slash").replace(":","colan").replace(";","semi-colan").replace("<","less-than-sign").replace(">","greater-than-sign").replace("=","equal-sign").replace("?","question-mark").replace("@","at-sign").replace("[","open-sq-brackets").replace("\\","back-slash").replace("]","close-sq-brackets").replace("^","caret").replace("_","underscore").replace("`","grave-accent").replace("{","open-curly-brace").replace("|","vertical-bar").replace("}", "close-curly-brace").replace("~", "tilde").replace("#", "hash").replace("•","bullet-sign").replace(" ","-")

    @staticmethod
    def get_top_from_dictionary(dictionary, min_prune_value = 1):
        items_to_remove = []
        for key, value in dictionary.items():
            if value < min_prune_value:
                items_to_remove.append(key)
        for item in items_to_remove:
            del dictionary[item]
        return dictionary

    @staticmethod
    def misspelled_word_list(CONFIG, word_tokens):
        misspelled_freq_dist = {}
        spell = SpellChecker()
        # words like I'll I'm are indicated as misspelt, this is fine though that means that person spells it in that particular manner and is beieng flagged
        # misspelled = spell.unknown(['somessthing', 'is', 'hapenning', 'here', '!', '👍', ',', ':)', 'somessthing'])
        # misspelled = spell.unknown(['hamzas', 'emails', "I'll", "I'm"])
        misspelled = spell.unknown(word_tokens)
        word_tokens_lowercase = []
        for w in word_tokens:
            word_tokens_lowercase.append(w.lower())
        for w in misspelled:
            misspelled_freq_dist[w] = word_tokens_lowercase.count(w)
        # sorted(misspelled_freq_dist.items(), key=lambda x: x[1], reverse=True)
        misspelled_freq_dist = CANLTK.get_top_from_dictionary(misspelled_freq_dist, CONFIG['MISSPELLED_MIN_PRUNE_VALUE'])
        return misspelled_freq_dist

    # Old method
    # line = f2.readline().rstrip('\n')
    # while line:
    #     generate_values(line)
    #     line = f2.readline().rstrip('\n')
    # f2.close()

    # def preprocessing_fn(inputs):
    #     modified_inputs = {}
    #     for key, value in inputs.items():
    #         modified_inputs[key] = tft.scale_to_0_1(value)
    #     return modified_inputs

    # def normalize_data(data):
    #       # Ignore the warnings
    #     fieldnames = data[0].keys()
    #     datatypes = {}
    #     for fn in fieldnames:
    #         datatypes[fn] = tf.io.FixedLenFeature([], tf.float32)
    #     raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec(datatypes))
    #     with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    #         transformed_dataset, transform_fn = (  # pylint: disable=unused-variable
    #             (data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
    #                 preprocessing_fn))

    #     transformed_data, transformed_metadata = transformed_dataset  # pylint: disable=unused-variable
    #     return transformed_data

    @staticmethod
    def normalize_data(data):
        raw = []
        for i in data:
            r2 = []
            for k,v in i.items():
                r2.append(v)
            raw.append(r2)

        # r_normalized = preprocessing.normalize(raw, norm='l2',axis=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        r_normalized = min_max_scaler.fit_transform(raw)

        normalized_list = []
        for val in r_normalized:
            normalized_dict = {}
            heading_names = list(data[0])
            for index, inner_val in enumerate(val):
                normalized_dict[heading_names[index]] = inner_val
            normalized_list.append(normalized_dict)
        return normalized_list

# main()
# write code to generate negative results for analysis using same params


# ADDITIONAL REFERENCES
# https://pc.net/emoticons/
# TODO do CBOW for additional accuracy
# TODO use LIWC API or maybe develop some of those features http://www.utpsyc.org/TAT/LIWCTATresults.php https://liwc.wpengine.com/compare-dictionaries/
# TODO https://github.com/LSYS/lexicalrichness https://sp1718.github.io/nltk.pdf
# TODO https://github.com/adeshpande3/LSTM-Sentiment-Analysis
# 

# TODO Setiment Analysis
# https://github.com/axelnine/Sentiment-Analysis
# https://github.com/shubhi-sareen/Sentiment-Analysis
# https://github.com/ian-nai/Simple-Sentiment-Analysis
# https://github.com/changhuixu/sentiment-analysis-using-python
# https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
# https://github.com/MohamedAfham/Twitter-Sentiment-Analysis-Supervised-Learning


# TODO Lexical richness
# https://pypi.org/project/lexicalrichness/

# TODO Typo and word extensions
# https://pypi.org/project/pytypo/
# https://stackoverflow.com/questions/20170022/elongated-word-check-in-sentence

# TODO british or  american enlgish
# https://datascience.stackexchange.com/questions/23236/tokenize-text-with-both-american-and-english-words
# https://stackoverflow.com/questions/42329766/python-nlp-british-english-vs-american-english 
# https://github.com/hyperreality/American-British-English-Translator