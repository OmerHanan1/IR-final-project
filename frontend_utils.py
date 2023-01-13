import re
import math
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def query_expansion(query):
    synonyms = {}
    hyponyms = {}
    processed_words = set()
    word_synsets = {}
    for word in query:
        if word in processed_words:
            continue
        if word not in word_synsets:
            word_synsets[word] = wordnet.synsets(word)
        for synset in word_synsets[word]:
            syn = synset.lemmas()[0].name().lower()
            if syn and syn != word:
                if word not in synonyms:
                    synonyms[word] = []
                synonyms[word].append(syn)
        processed_words.add(word)

    for word in query:
        if word in processed_words:
            continue
        if word not in word_synsets:
            word_synsets[word] = wordnet.synsets(word)
        for synset in word_synsets[word]:
            try:
                hyp = synset.hyponyms()[0].name()
                word_synset = wordnet.synset(hyp)
                hypernyms = word_synset.hypernyms()
                lst = (synset.name().split(".")[0].lower() for synset in hypernyms)
                for h in lst:
                    if h != word:
                        if word not in hyponyms:
                            hyponyms[word] = []
                        hyponyms[word].append(h)
            except IndexError:
                pass
        processed_words.add(word)
    for key, val in synonyms.items():
        query.extend([v for v in val if v not in query])
    for key, val in hyponyms.items():
        query.extend([v for v in val if v not in query])

    return query

def tokenize(text, STEMMING=False):
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?[\w,]?[\w.]?(?:['\-]?[\w,]?[\w])){0,24}""", re.UNICODE)
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)

    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

    tokens = query_expansion(tokens)

    if STEMMING:
        stemmer = PorterStemmer()
        list_of_tokens = [stemmer.stem(x) for x in tokens if x not in all_stopwords]
    else:
        list_of_tokens = [x for x in tokens if x not in all_stopwords]
  
    return list_of_tokens


def BM25(tokens, K, B, AVGDL, inverted_index, index_folder_url, DL, DL_LEN):
    
    doc_BM25_value = Counter()

    for token in tokens:

        # calc idf for specific token
        try:
          token_df = inverted_index.df[token]
        except:
            continue
        token_idf = math.log(DL_LEN/token_df,10)

        # loading posting list with (word, (doc_id, tf))
        posting_list = inverted_index.read_posting_list(token, index_folder_url)
        for page_id, word_freq in posting_list:
            #normalized tf (by the length of document)
            try:
                numerator = word_freq*(K+1)
                denominator = word_freq + K*(1-B + (B*DL[page_id])/AVGDL)
                doc_BM25_value[page_id] += token_idf*(numerator/denominator)
            except:
                pass
        
    sorted_doc_BM25_value = doc_BM25_value.most_common()
    return sorted_doc_BM25_value


def cossim(tokens, inverted_index, index_folder_url, DL, DL_LEN, NF):
    
    # get frequency of each token in query
    query_freq = Counter(tokens)

    numerator = Counter()
    query_denominator = 0
    weight_token_query = 0

    query_len = len(tokens)
    for token in tokens:

        # calc idf for specific token
        try:
          token_df = inverted_index.df[token]
        except:
            continue
        token_idf = math.log(DL_LEN/token_df,10)

        # calc query_token_tf
        tf_of_query_token = query_freq[token]/query_len
        weight_token_query = tf_of_query_token*token_idf
        query_denominator += math.pow(weight_token_query ,2)

        # loading posting list with (word, (doc_id, tf))
        posting_list = inverted_index.read_posting_list(token, index_folder_url)
        for page_id, word_freq in posting_list:
            #normalized tf (by the length of document)
            try:
                tf = (word_freq/DL[page_id])
                weight_word_page = tf*token_idf
                numerator[page_id] += weight_word_page*weight_token_query
            except:
                pass

    cosim = Counter()
    for page_id in numerator.keys():
      cosim[page_id] = numerator[page_id]/((math.sqrt(query_denominator)*NF[page_id]))
    
    sorted_doc_cossim_value = cosim.most_common()
    return sorted_doc_cossim_value


def get_binary_score(tokens, inverted_index, index_folder_url):

    # loading posting list with (word, (doc_id, tf))
    posting_lists = inverted_index.get_posting_lists(tokens, index_folder_url)

    tf_dict = {}
    for posting in posting_lists:
        for doc_id, _ in posting:
            if doc_id in tf_dict:
                tf_dict[doc_id] += 1
            else:
                tf_dict[doc_id] = 1

    list_of_docs = sorted([(doc_id, score) for doc_id, score in tf_dict.items()], key=lambda x: x[1], reverse=True)   
    return list_of_docs