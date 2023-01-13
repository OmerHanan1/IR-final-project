import re
import math
from collections import Counter
from nltk.corpus import stopwords

def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.
    Parameters:
    -----------
    text: string , represting the text to tokenize.
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?[\w,]?[\w.]?(?:['\-]?[\w,]?[\w])){0,24}""", re.UNICODE)
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
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
            numerator = word_freq*(K+1)
            denominator = word_freq + K*(1-B + (B*DL[page_id])/AVGDL)
            doc_BM25_value[page_id] += token_idf*(numerator/denominator)
        
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