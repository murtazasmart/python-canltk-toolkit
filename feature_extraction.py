# -*- coding: utf-8 -*-
# pylint: disable=unused-variable
import csv
import re
import string
import sys
from functools import reduce
import random
import os

import emoji
import emot
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, TweetTokenizer 
import arff
import tempfile
import json
import time

from sklearn import preprocessing
import CANLTK

from CANLTK import CANLTK

class FeatureExtraction:

  stops = stopwords.words('english')
  twitter_tokenizer = TweetTokenizer()

  # def count__each_most_common_words(string):
  #     d = {}
  #     freq_dist = FreqDist(wordTokens)
  #     for li in freq_dist.most_common(10):
  #         li

  @staticmethod
  def create_csv_file(file_name, dictionary):
      with open(file_name, mode='w', encoding="utf8", newline='') as csv_file:
          fieldnames = dictionary[0].keys()
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          writer.writeheader()
      #         for row in dictionary:
          writer.writerows(dictionary)

  @staticmethod
  def generate_values(CONFIG, line, data_dictionary, feature_table, static_feature_table, dynamic_feature_table, RESULT):
      static_d = {}
      dynamic_d = {}
      CONFIG['feature_count']
      CONFIG['feature_count'] = 0
      CONFIG['feature_count']+=1
      ss = ""
      ss.replace
      static_d["f" + str(CONFIG['feature_count']) + "-character-count"] = len(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-lowercase-count"] = CANLTK.n_lower_chars(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-uppercase-count"] = CANLTK.n_upper_chars(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-spaces-count"] = CANLTK.n_isspace_chars(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-vowels-count"] = CANLTK.n_vowels_chars(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-special-chars-count"] = CANLTK.n_special_chars(line)
      static_d.update(CANLTK.n_count_and_print_alphabets(CONFIG, line))
      static_d.update(CANLTK.n_count_and_print_special_chars(CONFIG, line))
          # contain bais for skin tone
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-emoticons-count"] = CANLTK.n_total_emoticons(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-emojis-count"] = CANLTK.n_total_emojis(line)

      wordTokens_with_emojis_emoticons = FeatureExtraction.twitter_tokenizer.tokenize(line)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-lexical-ttr"] = CANLTK.lexical_ttr(wordTokens_with_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-lexical-mtld"] = CANLTK.lexical_mtld(wordTokens_with_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-lexical-msttr"] = CANLTK.lexical_msttr(wordTokens_with_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-lexical-mattr"] = CANLTK.lexical_mattr(wordTokens_with_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-lexical-hdd"] = CANLTK.lexical_hdd(wordTokens_with_emojis_emoticons)

      # pylint: disable=unused-variable
      for key, value in data_dictionary['top_emoticons'].items():
          # TODO COUNT IS BEING DONE
          CONFIG['feature_count']+=1
          dynamic_d["f" + str(CONFIG['feature_count']) + "-emoticons-" + CANLTK.replace_keys(key)] = line.count(key)
          
      for key, value in data_dictionary['top_emojis'].items():
          # TODO COUNT IS BEING DONE
          CONFIG['feature_count']+=1
          # TODO removing emoji for tensorflow compatibility
          dynamic_d["f" + str(CONFIG['feature_count']) + "-emojis"] = line.count(key)
          # dynamic_d["f" + str(CONFIG['feature_count']) + "-emojis-" + replace_keys(key)] = line.count(key)

      # static_d.update(n_each_emoticons(line))
      # static_d.update(n_each_emojis(line))
    #   line = CANLTK.prune_emojis_emoticons(line)
      line_pruned_emojis_emoticons = CANLTK.prune_emojis_emoticons(line)
      # TODO emojis per word
      # TODO no of emojis per character
      # TODO CODE FOR character 5 and 6 grams ... dont understand

      # the regex here checks for all characters and apostrophes, might not behave well when it comes to long texts messages
          # tokenizer = RegexpTokenizer(r'[\w\']+')
          # wordTokens = tokenizer.tokenize(line)
      # this even cosiders ? as words and Okay👍👍🏿 as a single word
      #     print(word_tokenize(line))
      # this one is ideal breaks up I'll emojis etc, has been trained on twitter dataset
      wordTokens = FeatureExtraction.twitter_tokenizer.tokenize(line_pruned_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-word-count"] = len(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-word-len-less-than-6"] = CANLTK.n_long_words(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-word-len-less-than-equal-to-3"] = CANLTK.n_words_le_3(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-word-len-less-than-equal-to-2"] = CANLTK.n_words_le_2(wordTokens)
      # TODO average sentence length in terms of words
      # when deciding longer than 6, shorter than 2 and 3 apostrophes are considered as a character
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-word-len-avg"] = CANLTK.avg_words(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-misspelled-words"] = CANLTK.n_misspelled_words(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-avg-of-syllables-per-word"] = CANLTK.avg_syllables(wordTokens)
          # ratio of characters in words to N
          # repalced words
      #     print(n_punctuation(line))
      # word extensions - this can be covered to a certain extent with list of common misspelled words
      # both of these count punctuations used in emoticons too, need to remove them, need to cleanse emojis and send
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-punctuations-count"] = CANLTK.n_total_punctuations(line_pruned_emojis_emoticons)
      static_d.update(CANLTK.print_n_each_punctuation(CONFIG, line_pruned_emojis_emoticons))
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-total-unicode-punctuations"] = CANLTK.n_punctuation(line_pruned_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-function-words"] = CANLTK.n_function_words(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-context-words"] = CANLTK.n_context_words(wordTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-word-extensions"] = CANLTK.n_word_extensions(wordTokens)
      # TODO doesnt list each unicode - general punctioation and its frequency
      # TODO relative frequency of function word
      sentTokens = sent_tokenize(line_pruned_emojis_emoticons)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-sentences"] = len(sentTokens)
      # isnt this same as avg_words print("average number of character in a line ", )

      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-sentences-starting-with-lowercase"] = CANLTK.n_lowercase_sentences(sentTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-sentences-starting-with-uppercase"] = CANLTK.n_uppercase_sentences(sentTokens)
      CONFIG['feature_count']+=1
      static_d["f" + str(CONFIG['feature_count']) + "-grammar-mistakes"] = CANLTK.n_grammar_errors(line_pruned_emojis_emoticons)

      for w in data_dictionary['most_common']:
          CONFIG['feature_count']+=1
          dynamic_d["f" + str(CONFIG['feature_count']) + "-mc-" + CANLTK.replace_keys(w[0])] = CANLTK.n_words(w[0], wordTokens)

      for key, value in data_dictionary['misspelled_freq_dist'].items():
          CONFIG['feature_count']+=1
          dynamic_d["f" + str(CONFIG['feature_count']) + "-ms-misspelled-" + CANLTK.replace_keys(key)] = CANLTK.n_words(key, wordTokens)
      
      word_extensions = CANLTK.count_each_word_extension(data_dictionary['most_common_word_extensions'], wordTokens)
      for x in word_extensions:
          CONFIG['feature_count']+=1
          dynamic_d["f" + str(CONFIG['feature_count']) + "-mc-extensions-" + CANLTK.replace_keys(x["word"])] = x["count"]

      bigrams_list_of_dict = CANLTK.count_each_ngrams(wordTokens_with_emojis_emoticons, data_dictionary['most_common_bigrams'])
      bigram_count = 0
      for x in bigrams_list_of_dict:
          CONFIG['feature_count']+=1
          dynamic_d["f" + str(CONFIG['feature_count']) + "-mc-bigrams-" + str(bigram_count)] = x["count"]
          bigram_count = bigram_count + 1

      trigrams_list_of_dict = CANLTK.count_each_ngrams(wordTokens_with_emojis_emoticons, data_dictionary['most_common_trigrams'])
      trigram_count = 0
      for x in trigrams_list_of_dict:
          CONFIG['feature_count']+=1
          dynamic_d["f" + str(CONFIG['feature_count']) + "-mc-trigrams-" + str(trigram_count)] = x["count"]
          trigram_count = trigram_count + 1

      # d["result"] = RESULT
      full_d = {}
      full_d.update(static_d)
      full_d.update(dynamic_d)
      static_d["result"] = RESULT
      dynamic_d["result"] = RESULT
      full_d["result"] = RESULT
      static_feature_table.append(static_d)
      dynamic_feature_table.append(dynamic_d)
      feature_table.append(full_d)

      return feature_table, static_feature_table, dynamic_feature_table
      # FOR PARA
      # hapax legomena and dis legomena, a word that only appeaers once in the text, word that appears only twice in the text. This can only be done in a group of texts
      # vocabulary richness (number of unique words)
      # this data can be taken in for PARAs and

      # REDUNDANT
      # average number of words in a block 
      # average number of sentences in a block - not needed
      # number of paragraphs
      # average number of characters in a block

  # returns a list of tuples with word and count i.e. [('okay',2)]
  @staticmethod
  def full_chat_analysis(CONFIG):
    file = open(CONFIG['INPUT_FILE_NAME'], encoding="utf8")
    file_line = file.readline().rstrip('\n')
    file_string = ""
    while file_line:
      file_line = file_line.replace('\u200e', "")
      file_line = file_line.replace('\u200b', "")
      file_line = file_line.replace('\u200d', "")
      file_string = file_string + file_line + '. '
      file_line = file.readline().rstrip('\n')
    file.close()
    # Counting each emojis and emoticon
    emoticons_dictionary = CANLTK.n_each_emoticons(file_string)
    emojis_dictionary = CANLTK.n_each_emojis(file_string)
    word_tokens_with_emojis_emoticons = FeatureExtraction.twitter_tokenizer.tokenize(file_string)
    most_common_bigrams_dictionary = CANLTK.most_common_bigrams(CONFIG, word_tokens_with_emojis_emoticons)
    most_common_trigrams_dictionary = CANLTK.most_common_trigrams(CONFIG, word_tokens_with_emojis_emoticons)
    # Getting top emojis and emoticons
    top_emoticons = CANLTK.get_top_from_dictionary(emoticons_dictionary, CONFIG["EMOJI_EMOTICON_MIN_PRUNE_VALUE"])
    top_emojis = CANLTK.get_top_from_dictionary(emojis_dictionary, CONFIG["EMOJI_EMOTICON_MIN_PRUNE_VALUE"])
    # Removing emojis and emoticons from file string
    file_string = CANLTK.prune_emojis_emoticons(file_string)
    word_tokens = FeatureExtraction.twitter_tokenizer.tokenize(file_string)
    word_tokens = CANLTK.prune_punctuations_special_characters(word_tokens)
    most_common_word_extensions_dictionary = CANLTK.most_common_word_extensions(CONFIG, word_tokens)
    word_tokens = CANLTK.prune_function_words(word_tokens)
    # // add most often words extended
    freq_dist = FreqDist(word_tokens)
    misspelled_freq_dist = CANLTK.misspelled_word_list(CONFIG, word_tokens)
    data_dictionary = {
      'most_common': freq_dist.most_common(CONFIG['MOST_COMMON_WORDS']),
      'top_emoticons': top_emoticons,
      'top_emojis': top_emojis,
      'misspelled_freq_dist': misspelled_freq_dist,
      "most_common_word_extensions": most_common_word_extensions_dictionary,
      "most_common_bigrams" : most_common_bigrams_dictionary,
      "most_common_trigrams": most_common_trigrams_dictionary
    }
    jsond = json.dumps(data_dictionary)
    f = open(CONFIG['FEATURE_SET_FILE_NAME'],"w")
    f.write(jsond)
    f.close()
    return data_dictionary

  @staticmethod
  def create_arff_output(CONFIG, feature_table):
      arff_attributes = []
      for attr in list(feature_table[0]):
          if attr == "result":
              arff_attributes.append(("class", u'REAL'))
          else:
              arff_attributes.append((attr, u'REAL'))
      arff_data = {
          u'attributes': arff_attributes,
          u'data': [
          ],
          u'description': u'Murtaza MSc',
          u'relation': u'weather'
      }
      for li in feature_table:
          arff_data['data'].append(li.values())
      # file = open( "chat4-HamzaNajmudeen-AbbasJafferjee.arff", 'w')
      file = open(CONFIG['ARFF_OUTPUT_FILE_NAME'], 'w', encoding="utf-8")
      arff.dump(arff_data, file)
      file.close()

  @staticmethod
  def negative_dataset_get_relevant_data(full_dictionary, template_dictionary):
      final_dictionary = {}
      for key, value in full_dictionary.items():
          if key in template_dictionary:
              final_dictionary[key] = value
      return final_dictionary
      
  # def negative_dataset_chat_analysis(data_dictionary):
  #     file = open(NEGATIVE_DATA_INPUT_FILE_NAME, encoding="utf8")
  #     file_line = file.readline().rstrip('\n')
  #     file_string = ""
  #     while file_line:
  #         file_string = file_string + file_line + '. '
  #         file_line = file.readline().rstrip('\n')
  #     file.close()
  #     emoticons_dictionary = n_each_emoticons(file_string)
  #     emojis_dictionary = n_each_emojis(file_string)
  #     # NOTE get same emoji list as postive set
  #     final_emoticons_dictionary = negative_dataset_get_relevant_data(emoticons_dictionary, data_dictionary['top_emoticons'])
  #     final_emojis_dictionary = negative_dataset_get_relevant_data(emojis_dictionary, data_dictionary['top_emojis'])
  #     file_string = prune_emojis_emoticons(file_string)
  #     # word_tokens = word_tokenize(file_string)
  #     word_tokens = twitter_tokenizer.tokenize(file_string)
  #     # print(wordTokens)
  #     word_tokens = prune_punctuations_special_characters(word_tokens)
  #     word_tokens = prune_function_words(word_tokens)
  #     # print(stops)
  #     # print(wordTokens)
  #     misspelled_freq_dist = misspelled_word_list(word_tokens)
  #     final_misspelled_freq_dist = negative_dataset_get_relevant_data(misspelled_freq_dist, data_dictionary['misspelled_freq_dist'])
  #     data_dictionary = {
  #         'most_common': data_dictionary['most_common'],
  #         'top_emoticons': final_emoticons_dictionary,
  #         'top_emojis': final_emojis_dictionary,
  #         'misspelled_freq_dist': final_misspelled_freq_dist
  #     }
  #     return data_dictionary


  @staticmethod
  def create_inverse_dataset(CONFIG, chat_count):
      directory = r'D:/MSc/Chat Parser Script/chat-data/processed-chat'
      dir_list = os.listdir(directory)
      l = []
      for filename in dir_list:
          if CONFIG['INPUT_FILE_NAME'].find(filename) == -1:
              s = open(os.path.join(directory, filename),"r", encoding="utf8")
              m = s.readlines()
              n = []
              iterations = round(chat_count / (len(dir_list) - 2))
              if iterations == 0:
                  iterations = 1
              # pylint: disable=unused-variable
              for i in range(iterations):
                  if len(n) == len(m):
                      break
                  ranNumber = random.randint(0, len(m) - 1)
                  # while n.count(ranNumber) != 0 || len(m[ranNumber])<15:
                  # If we wanna add a min limit to the strings
                  while n.count(ranNumber) != 0:
                      if len(n) == len(m):
                          break
                      ranNumber = random.randint(0, len(m) - 1)
                  n.append(ranNumber)
                  l.append(m[ranNumber])
      return l

  @staticmethod
  def create_full_inverse_dataset(CONFIG):
      directory = r'D:/MSc/Chat Parser Script/chat-data/processed-chat'
      dir_list = os.listdir(directory)
      l = []
      for filename in dir_list:
          if CONFIG['NPUT_FILE_NAME'].find(filename) == -1:
              s = open(os.path.join(directory, filename),"r", encoding="utf8")
              m = s.readlines()
              n = []
              # pylint: disable=unused-variable
              for i in range(CONFIG['FULL_TEST_COUNT_PER_CHAT']):
                  if len(n) == len(m):
                      break
                  ranNumber = random.randint(0, len(m) - 1)
                  # while n.count(ranNumber) != 0 || len(m[ranNumber])<15:
                  # If we wanna add a min limit to the strings
                  while n.count(ranNumber) != 0:
                      if len(n) == len(m):
                          break
                      ranNumber = random.randint(0, len(m) - 1)
                  l.append(m[ranNumber])
                  n.append(ranNumber)
      return l

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

  @staticmethod
  def run_extraction(CONFIG):
      RESULT = 1
      f1 = open(CONFIG['INPUT_FILE_NAME'], encoding="utf8")
      data_dictionary = FeatureExtraction.full_chat_analysis(CONFIG)
      # TODO hapax legomena needs to be researched more, check https://kite.com/python/docs/nltk.FreqDist

      # FOR POSITIVE DATASET
      line = f1.readline().rstrip('\n')
      feature_table = []
      static_feature_table = []
      dynamic_feature_table = []
      chat_count = 0
      while line:
          # line = line.strip('\u200e')
          feature_table, static_feature_table, dynamic_feature_table = FeatureExtraction.generate_values(CONFIG, line, data_dictionary, feature_table, static_feature_table, dynamic_feature_table, RESULT)
          line = f1.readline().rstrip('\n')
          chat_count += 1
      f1.close()
      POSITIVE_CHAT_COUNT = len(feature_table)

      # FOR NEGATIVE DATASET
      # negative_dataset_chat_analysis(data_dictionary)
      # TODO check if already a inverse file exists
      RESULT = 0
      inverse_chat_lines = FeatureExtraction.create_inverse_dataset(CONFIG, chat_count)

      inverse_chats_file = open(CONFIG['INVERSE_CHATS_FILE_NAME'], "w", encoding="utf-8")
      for i in inverse_chat_lines:
          # line = line.strip('\u200e')
          feature_table, static_feature_table, dynamic_feature_table = FeatureExtraction.generate_values(CONFIG, i, data_dictionary, feature_table, static_feature_table, dynamic_feature_table, RESULT)
          inverse_chats_file.write(i)
      inverse_chats_file.close()
      NEGATIVE_CHAT_COUNT = len(feature_table) - POSITIVE_CHAT_COUNT

      normalizedData = FeatureExtraction.normalize_data(feature_table)
      staticNormalizedData = FeatureExtraction.normalize_data(static_feature_table)
      dynamicNormalizedData = FeatureExtraction.normalize_data(dynamic_feature_table)

      # TODO need to first split negative and postive data set equally and then carry out shuffling

      TOTAL_CHAT_COUNT = len(feature_table)
      POSITIVE_SPLIT_INDEX = int(POSITIVE_CHAT_COUNT*(1-CONFIG['TEST_SPLIT']))
      NEGATIVE_SPLIT_INDEX = int(NEGATIVE_CHAT_COUNT*(1-CONFIG['TEST_SPLIT']))

      train_data_positive = normalizedData[0:POSITIVE_SPLIT_INDEX]
      train_data_negative = normalizedData[POSITIVE_CHAT_COUNT:(POSITIVE_CHAT_COUNT + NEGATIVE_SPLIT_INDEX)]
      test_data_positive = normalizedData[POSITIVE_SPLIT_INDEX:POSITIVE_CHAT_COUNT]
      test_data_negative = normalizedData[(POSITIVE_CHAT_COUNT + NEGATIVE_SPLIT_INDEX):TOTAL_CHAT_COUNT]

      normalize_data_shuffle = random.sample(normalizedData, len(normalizedData))

      train_data = train_data_positive + train_data_negative
      test_data = test_data_positive + test_data_negative

      train_data_shuffle = random.sample(train_data, len(train_data))
      test_data_shuffle = random.sample(test_data, len(test_data))

      # create_arff_output(feature_table)
      FeatureExtraction.create_csv_file(CONFIG['CSV_OUTPUT_FILE_NAME'], feature_table)
      FeatureExtraction.create_csv_file(CONFIG['NORMALIZED_FULL_CSV_OUTPUT_FILE_NAME'], normalizedData)
      FeatureExtraction.create_csv_file(CONFIG['NORMALIZED_STATIC_ONLY_CSV_OUTPUT_FILE_NAME'], staticNormalizedData)
      FeatureExtraction.create_csv_file(CONFIG['NORMALIZED_DYNAMIC_ONLY_CSV_OUTPUT_FILE_NAME'], dynamicNormalizedData)
      FeatureExtraction.create_csv_file(CONFIG['SHUFFLED_NORMALIZED_CSV_OUTPUT_FILE_NAME'], normalize_data_shuffle)
      FeatureExtraction.create_csv_file(CONFIG['TRAIN_CSV_OUTPUT_FILE_NAME'], train_data_shuffle)
      FeatureExtraction.create_csv_file(CONFIG['TEST_CSV_OUTPUT_FILE_NAME'], test_data_shuffle)


      # For full negative dataset for testing, evaluating purposes
      # feature_table = []
      # static_feature_table = []
      # dynamic_feature_table = []
      # full_inverse_chat_lines = create_full_inverse_dataset()

      # full_inverse_chats_file = open(FULL_INVERSE_CHATS_FILE_NAME, "w", encoding="utf-8")
      # for i in full_inverse_chat_lines:
      #     # line = line.strip('\u200e')
      #     generate_values(i, data_dictionary)
      #     full_inverse_chats_file.write(i)
      # full_inverse_chats_file.close()

      # normalizedData = normalize_data(feature_table)
      # create_csv_file(FULL_TEST_CSV_OUTPUT_FILE_NAME, normalizedData)


  # feature_count = 0
  # TEST_SPLIT = 0.2
  # MOST_COMMON_WORDS = 25
  # MISSPELLED_MIN_PRUNE_VALUE = 25
  # FULL_TEST_COUNT_PER_CHAT = 100

  @staticmethod
  def set_variables(BASE_NAME):
      CONFIG = {}
      CONFIG['FILE_NAME'] = BASE_NAME + '.txt'
      CONFIG['BASE_OUTPUT_FOLDER'] = 'D:/MSc/Chat Parser Script/chat-data/extracted-features/'
      CONFIG['BASE_INPUT_FOLDER'] = 'D:/MSc/Chat Parser Script/chat-data/processed-chat/'
      CONFIG['INPUT_FILE_NAME'] = CONFIG['BASE_INPUT_FOLDER'] + BASE_NAME + '.txt'
      CONFIG['ARFF_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '.arff'
      # NEGATIVE_DATA_INPUT_FILE_NAME = BASE_OUTPUT_FOLDER + 'chat1-MustafaAbid-MurtazaAn.txt'
      CONFIG['CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-partial.csv'
      CONFIG['NORMALIZED_STATIC_ONLY_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-static-normalized.csv'
      CONFIG['NORMALIZED_DYNAMIC_ONLY_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-dynamic-normalized.csv'
      CONFIG['NORMALIZED_FULL_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-full-normalized.csv'
      CONFIG['FEATURE_SET_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-feature-set.json'
      CONFIG['SHUFFLED_NORMALIZED_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-normalized-shuffled.csv'
      CONFIG['TRAIN_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-normalized-train-set.csv'
      CONFIG['TEST_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-normalized-test-set.csv'
      CONFIG['FULL_TEST_CSV_OUTPUT_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-normalized-full-test-set.csv'
      CONFIG['INVERSE_CHATS_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-inverse.txt'
      CONFIG['FULL_INVERSE_CHATS_FILE_NAME'] = CONFIG['BASE_OUTPUT_FOLDER'] + BASE_NAME + '-full-inverse.txt'
      CONFIG['TEST_SPLIT'] = 0.2
      CONFIG['MOST_COMMON_WORDS'] = 25
      CONFIG['MISSPELLED_MIN_PRUNE_VALUE'] = 25
      CONFIG['FULL_TEST_COUNT_PER_CHAT'] = 100
      CONFIG['BIGRAMS_MIN_PRUNE_VALUE'] = 5
      CONFIG['TRIGRAMS_MIN_PRUNE_VALUE'] = 5
      CONFIG['NO_OF_MOST_FREQ_WORD_EXTENSIONS'] = 25
      CONFIG['EMOJI_EMOTICON_MIN_PRUNE_VALUE'] = 1
      CONFIG['feature_count'] = 0
      return CONFIG

  @staticmethod
  def feature_extraction(runCompleteItertaion=False, BASE_NAME="chat3-AbbasJafferjee-HamzaNajmudeen"):
      if runCompleteItertaion:
          for root, dirs, files in os.walk(".\\chat-data\\processed-chat"):
              # MULTI FILE CODE
              for filename in files:
                  print(filename[:-4])
                  BASE_NAME = filename[:-4]
                  if not os.path.exists("D:/MSc/Chat Parser Script/chat-data/extracted-features/" + BASE_NAME + "-normalized-train-set.csv"):
                    timeStats = {}
                    startTime = time.time()
                    CONFIG = FeatureExtraction.set_variables(BASE_NAME)
                    FeatureExtraction.run_extraction(CONFIG)
                    endTime = time.time()
                    timeStats["featureExtraction"] = endTime - startTime
                    jsond = json.dumps(timeStats)
                    f = open("D:/MSc/Chat Parser Script/chat-data/timings/" + BASE_NAME + "feature-extraction-time-stats.json", "w")
                    f.write(jsond)
                    f.close()
                    print("Completed extratction for " + BASE_NAME)
                  else:
                    print("Skipping ", BASE_NAME)
              print(len(files))
      else:
          # SINLE FILE CODE
          CONFIG = FeatureExtraction.set_variables(BASE_NAME)
          FeatureExtraction.run_extraction(CONFIG)

# nameList = [
#     "chat115-GehanCooray-SamaliL",
#     "chat116-SamaliL-GehanCooray"
# ]
# for l in nameList:
#     FeatureExtraction.feature_extraction(BASE_NAME=l)
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