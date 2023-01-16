from flask import Flask, request, jsonify, render_template
import gzip
import pandas as pd
import pickle
from inverted_index_gcp import *
from frontend_utils import *

INVERTED_INDEX_FILE_NAME = "index"

POSTINGS_TEXT_OLD_FOLDER_URL = "postings_gcp_text_old"
POSTINGS_ANCHOR_OLD_FOLDER_URL = "postings_gcp_anchor_old"
POSTINGS_TITLE_OLD_FOLDER_URL = "postings_gcp_title_old"
OLD_DL_PATH = "old_dl/dl.pkl"
OLD_NF_PATH = "old_nf/nf.pkl"

POSTINGS_TEXT_FOLDER_URL = "postings_gcp_text"
POSTINGS_ANCHOR_FOLDER_URL = "postings_gcp_anchor"
POSTINGS_TITLE_FOLDER_URL = "postings_gcp_title"
POSTINGS_TEXT_STEMMED_FOLDER_URL = "postings_gcp_text_stemmed"
POSTINGS_ANCHOR_STEMMED_FOLDER_URL = "postings_gcp_anchor_stemmed"
POSTINGS_TITLE_STEMMED_FOLDER_URL = "postings_gcp_title_stemmed"
DL_PATH = "dl/dl.pkl"
NF_PATH = "nf/nf.pkl"

PAGE_RANK_URL = "pr/part-00000-8b293cd5-fd79-47e7-a641-3d067da0c2b0-c000.csv.gz"
PAGE_VIEW_URL = "pv/pageview_pageviews-202108-user.pkl"
DT_PATH = "dt/dt.pkl"


# open files (inverted indexes etc...)
inverted_index_body_old = InvertedIndex.read_index(POSTINGS_TEXT_OLD_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_anchor_old = InvertedIndex.read_index(POSTINGS_ANCHOR_OLD_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_title_old = InvertedIndex.read_index(POSTINGS_TITLE_OLD_FOLDER_URL, INVERTED_INDEX_FILE_NAME)

inverted_index_body = InvertedIndex.read_index(POSTINGS_TEXT_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_anchor = InvertedIndex.read_index(POSTINGS_ANCHOR_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_title = InvertedIndex.read_index(POSTINGS_TITLE_FOLDER_URL, INVERTED_INDEX_FILE_NAME)

inverted_index_body_stemmed = InvertedIndex.read_index(POSTINGS_TEXT_STEMMED_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_anchor_stemmed = InvertedIndex.read_index(POSTINGS_ANCHOR_STEMMED_FOLDER_URL, INVERTED_INDEX_FILE_NAME)
inverted_index_title_stemmed = InvertedIndex.read_index(POSTINGS_TITLE_STEMMED_FOLDER_URL, INVERTED_INDEX_FILE_NAME)

with open(DL_PATH, 'rb') as f:
    DL = pickle.load(f)
    DL_LEN = len(DL)

with open(OLD_DL_PATH, 'rb') as f:
    OLD_DL = pickle.load(f)
    OLD_DL_LEN = len(DL)

with open(NF_PATH, 'rb') as f:
    NF = pickle.load(f)

with open(OLD_NF_PATH, 'rb') as f:
    OLD_NF = pickle.load(f)

with open(DT_PATH, 'rb') as f:
    DT = pickle.load(f)

with open(PAGE_VIEW_URL, 'rb') as f:
    page_view = pickle.load(f)
    max_pv_value = max(page_view.values())
    norm_page_view = {doc_id: view/max_pv_value for doc_id, view in page_view.items()}

with gzip.open(PAGE_RANK_URL) as f:
    page_rank = pd.read_csv(f, header=None, index_col=0).squeeze("columns").to_dict()
    max_pr_value = max(page_rank.values())
    norm_page_rank = {doc_id: rank/max_pr_value for doc_id, rank in page_rank.items()}

# flask app
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route('/')
def show_shmoogle():
    return render_template('shmoogle.html')


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # const bool  
    QUERYEXP = False 
    STEMMING = True
    COSSIM = False

    K = 1.2
    B = 0.75
    AVGDL = 341.0890174848911
    LIMIT_DOCS = 4
    Wa = 0.95
    Wb = 1.1
    Wt = 0.95
    Wpv = 1
    Wpr = 1
    if "?" in query:
      Wb = Wb*1.2
      Wa = Wa*1.2
      Wpv = Wpv*0.8
      Wpr = Wpr*0.8
      


    # tokenizing the query
    tokens = tokenize(query, STEMMING, QUERYEXP)
    
    clac_score = Counter()

    if STEMMING:
        inverted_index_b = inverted_index_body_stemmed
        inverted_index_t = inverted_index_title_stemmed
        inverted_index_a = inverted_index_anchor_stemmed
        inverted_index_b_folder_url = POSTINGS_TEXT_STEMMED_FOLDER_URL
        inverted_index_t_folder_url = POSTINGS_TITLE_STEMMED_FOLDER_URL
        inverted_index_a_folder_url = POSTINGS_ANCHOR_STEMMED_FOLDER_URL
    else:
        inverted_index_b = inverted_index_body
        inverted_index_t = inverted_index_title
        inverted_index_a = inverted_index_anchor
        inverted_index_b_folder_url = POSTINGS_TEXT_FOLDER_URL
        inverted_index_t_folder_url = POSTINGS_TEXT_FOLDER_URL
        inverted_index_a_folder_url = POSTINGS_TEXT_FOLDER_URL

    if COSSIM:
        sorted_doc_text_score_pairs = cossim(tokens, inverted_index_b, inverted_index_b_folder_url, DL, DL_LEN, NF)[:500]
    else:
        sorted_doc_text_score_pairs = BM25(tokens, K, B, AVGDL, inverted_index_b, inverted_index_b_folder_url, DL, DL_LEN)[:500]
    
    max_value_score_body = sorted_doc_text_score_pairs[0][1]
    sorted_doc_text_score_pairs_norm = [(x[0], x[1]/max_value_score_body) for x in sorted_doc_text_score_pairs]

    sorted_doc_title_score_pairs = get_binary_score(tokens, inverted_index_t, inverted_index_t_folder_url)[:500]
    sorted_doc_title_score_pairs_norm = [(x[0], x[1]/len(tokens)) for x in sorted_doc_title_score_pairs]

    sorted_doc_anchor_score_pairs = get_power_score(tokens, inverted_index_a, inverted_index_a_folder_url)[:500]
    max_value_score_anchor = sorted_doc_anchor_score_pairs[0][1]
    sorted_doc_anchor_score_pairs_norm = [(x[0], x[1]/max_value_score_anchor) for x in sorted_doc_anchor_score_pairs]

    for doc_id, score in sorted_doc_text_score_pairs_norm:
        if doc_id in clac_score:
            clac_score[doc_id] += score*Wb
        else:
            clac_score[doc_id] = score*Wb

    for doc_id, score in sorted_doc_title_score_pairs_norm:
        if doc_id in clac_score:
            clac_score[doc_id] += score*Wt
        else:
            clac_score[doc_id] = score*Wt
    
    for doc_id, score in sorted_doc_anchor_score_pairs_norm:
        if doc_id in clac_score:
            clac_score[doc_id] += score*Wa
        else:
            clac_score[doc_id] = score*Wa
    
    # add page view
    for page_id in clac_score:
        try:
            clac_score[page_id] += norm_page_view[page_id]*Wpv
        except:
            pass  

    # add page rank
    for page_id in clac_score:
        try:
            clac_score[page_id] += norm_page_rank[page_id]*Wpr
        except:
            pass  

    sorted_clac_score = clac_score.most_common()

    # take first 100 
    best = sorted_clac_score[:100]

    # clac std and mean 
    xs = [x[1] for x in best]
    mean = sum(xs) / len(xs)
    var  = sum(pow(x-mean,2) for x in xs) / len(xs)
    std  = math.sqrt(var)

    # filter out results that are below (mean + 1.15*std)
    best = [best[i] for i in range(len(best)) if (best[i][1] > (mean + 1.15*std)) or (i < LIMIT_DOCS)]

    # take page titles according to id
    for doc_id, _ in best:
        try:
            res.append((doc_id, DT[doc_id]))
        except:
            pass       
    
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # tokenizing the query
    tokens = old_tokenize(query)

    # cossim
    sorted_doc_score_pairs = cossim(tokens, inverted_index_body_old, POSTINGS_TEXT_OLD_FOLDER_URL, OLD_DL, OLD_DL_LEN, OLD_NF)
    
    # take first 100 
    best = sorted_doc_score_pairs[:100]

    # take page titles according to id
    res = [(x[0], DT[x[0]]) for x in best]

    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # tokenizing
    tokens = old_tokenize(query)

    # get number of query tokens in doc_title
    list_of_docs = get_binary_score(tokens, inverted_index_title_old, POSTINGS_TITLE_OLD_FOLDER_URL)

    # generate doc_title for each doc_id
    for doc_id, _ in list_of_docs:
        try:
            res.append((doc_id, DT[doc_id]))
        except:
            pass   
    
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)

    # tokenizing
    tokens = old_tokenize(query)

    # get number of query tokens in doc_anchor_text
    list_of_docs = get_binary_score(tokens, inverted_index_anchor_old, POSTINGS_ANCHOR_OLD_FOLDER_URL)

    # generate doc_title for each doc_id
    for doc_id, _ in list_of_docs:
        try:
            res.append((doc_id, DT[doc_id]))
        except:
            pass   

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)

    for wiki_id in wiki_ids:
      try:
        res.append(page_rank[wiki_id])
      except:
        res.append(None)

    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)

    for wiki_id in wiki_ids:
      try:
        res.append(page_view[wiki_id])
      except:
        res.append(None)

    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
