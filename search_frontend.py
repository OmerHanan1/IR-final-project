from flask import Flask, request, jsonify, render_template
import gzip
import pandas as pd
import pickle
from inverted_index_gcp import *
from frontend_utils import *

INDEX_FILE = "index"
POSTINGS_GCP_TEXT_INDEX_FOLDER_URL = "postings_gcp_text"
POSTINGS_GCP_ANCHOR_INDEX_FOLDER_URL = "postings_gcp_anchor"
POSTINGS_GCP_TITLE_INDEX_FOLDER_URL = "postings_gcp_title"
POSTINGS_GCP_TEXT_STEMMED_INDEX_FOLDER_URL = "postings_gcp_text_stemmed"
POSTINGS_GCP_ANCHOR_STEMMED_INDEX_FOLDER_URL = "postings_gcp_anchor_stemmed"
POSTINGS_GCP_TITLE_STEMMED_INDEX_FOLDER_URL = "postings_gcp_title_stemmed"
PAGE_RANK_URL = "pr/pr.csv.gz"
PAGE_VIEW_URL = "pv/pv.pkl"
DT_PATH = "dt/dt.pkl"
DL_PATH = "dl/dl.pkl"
NF_PATH = "nf/nf.pkl"


# open files (inverted indexes etc...)
inverted_index_body = InvertedIndex.read_index(POSTINGS_GCP_TEXT_INDEX_FOLDER_URL, INDEX_FILE)
inverted_index_anchor = InvertedIndex.read_index(POSTINGS_GCP_ANCHOR_INDEX_FOLDER_URL, INDEX_FILE)
inverted_index_title = InvertedIndex.read_index(POSTINGS_GCP_TITLE_INDEX_FOLDER_URL, INDEX_FILE)
inverted_index_body_stemmed = InvertedIndex.read_index(POSTINGS_GCP_TEXT_STEMMED_INDEX_FOLDER_URL, INDEX_FILE)
inverted_index_anchor_stemmed = InvertedIndex.read_index(POSTINGS_GCP_ANCHOR_STEMMED_INDEX_FOLDER_URL, INDEX_FILE)
inverted_index_title_stemmed = InvertedIndex.read_index(POSTINGS_GCP_TITLE_STEMMED_INDEX_FOLDER_URL, INDEX_FILE)

with open(DL_PATH, 'rb') as f:
    DL = pickle.load(f)
    DL_LEN = len(DL)

with open(DT_PATH, 'rb') as f:
    DT = pickle.load(f)

with open(NF_PATH, 'rb') as f:
    NF = pickle.load(f)

with open(PAGE_VIEW_URL, 'rb') as f:
    page_view = pickle.load(f)

with gzip.open(PAGE_RANK_URL) as f:
    page_rank = pd.read_csv(f, header=None, index_col=0).squeeze("columns").to_dict()
    max_pr_value = max(page_rank.values())
    page_rank = {doc_id: rank/max_pr_value for doc_id, rank in page_rank.items()}

# flask app
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route('/')
# def show_banana():
#     return render_template('banana.html')
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

    # tokenizing the query
    tokens = tokenize(query, STEMMING, QUERYEXP)
    
    print(f"{'QUERYEXP' if QUERYEXP else 'NO QUERYEXP'} ############### {'STEMMING' if STEMMING else 'NO STEMMING'} ############### {'COSSIM' if COSSIM else 'BM25'}")

    if STEMMING:
        inverted_index = inverted_index_body_stemmed
        inverted_index_folder_url = POSTINGS_GCP_TEXT_STEMMED_INDEX_FOLDER_URL
    else:
        inverted_index = inverted_index_body
        inverted_index_folder_url = POSTINGS_GCP_TEXT_INDEX_FOLDER_URL

    if COSSIM:
        sorted_doc_score_pairs = cossim(tokens, inverted_index, inverted_index_folder_url, DL, DL_LEN, NF)
    else:
        sorted_doc_score_pairs = BM25(tokens, K, B, AVGDL, inverted_index, inverted_index_folder_url, DL, DL_LEN)
    
    # take first 100 
    best = sorted_doc_score_pairs[:100]

    # take page titles according to id
    res = [(x[0], DT[x[0]]) for x in best]
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
    tokens = tokenize(query)

    # cossim
    sorted_doc_score_pairs = cossim(tokens, inverted_index_body, POSTINGS_GCP_TEXT_INDEX_FOLDER_URL, DL, DL_LEN, NF)
    
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
    tokens = tokenize(query)

    # get number of query tokens in doc_title
    list_of_docs = get_binary_score(tokens, inverted_index_title, POSTINGS_GCP_TITLE_INDEX_FOLDER_URL)

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
    tokens = tokenize(query)

    # get number of query tokens in doc_anchor_text
    list_of_docs = get_binary_score(tokens, inverted_index_anchor, POSTINGS_GCP_ANCHOR_INDEX_FOLDER_URL)

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
        res.append(page_rank[wiki_id])
      except:
        res.append(None)

    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
