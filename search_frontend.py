from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
import re
import gzip
import os
import pickle
import pandas as pd
from inverted_index_gcp import *

POSTINGS_GCP_INDEX_URL = "postings_gcp_anchor_training/index.pkl"
PAGE_RANK_CSV_URL = "pr_part-00000-7cc9c080-88ee-451f-9087-06bc3a940d5e-c000.csv.gz"
PAGE_VIEW_DICT = "pageview_pageviews-202108-user.pkl"

# read posting list function
def read(index, token, path):

    BLOCK_SIZE = 1999998
    TUPLE_SIZE = 6

    # anchor index tuple size is 8 because the tuple consist of 2 doc_ids (Integer 4 bytes each)
    if "links" in path:
        TUPLE_SIZE = 8

    # Reading bin file
    n_bytes = index.df[token] * TUPLE_SIZE
    open_files = {}
    b = []
    for f_name, offset in index.posting_locs[token]:
        if f_name not in open_files:
            open_files[f_name] = open(os.path.join(path, f_name), 'rb')
        f = open_files[f_name]
        f.seek(offset)
        n_read = min(n_bytes, BLOCK_SIZE - offset)
        b.append(f.read(n_read))
        n_bytes -= n_read
    b = b''.join(b)

    for f in open_files.values():
        f.close()

    # Parsing bin file
    posting_list = []
    for i in range(index.df[token]):
        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
        posting_list.append((doc_id, tf))

    return posting_list


with open(POSTINGS_GCP_INDEX_URL, 'rb') as f:
    inverted_index_anchor = pickle.load(f)

with open(PAGE_VIEW_DICT, 'rb') as f:
    page_view = pickle.load(f)

with gzip.open(PAGE_RANK_CSV_URL) as f:
    page_rank = pd.read_csv(f, header=None, index_col=0).squeeze("columns").to_dict()
    max_pr_value = max(page_rank.values())
    page_rank = {doc_id: rank/max_pr_value for doc_id, rank in page_rank.items()}

# flask app
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)

@app.route('/')
# def show_banana():
#     return render_template('banana.html')
# def show_beautiful():
#     return render_template('beautiful.html')
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
    # BEGIN SOLUTION
    res.append("AMIR")
    # END SOLUTION
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
    # BEGIN SOLUTION

    # END SOLUTION
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
    # BEGIN SOLUTION

    # END SOLUTION
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
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens = [t for t in tokens if t not in all_stopwords]

    tf_dict = {}
    for token in tokens:
        added = []
        if token not in inverted_index_anchor.df:
            continue

        # loading posting list with (word, (doc_id_from, doc_id_dest))
        posting = read(inverted_index_anchor, token, "postings_gcp_anchor_training/")
        for _, doc_id_dest in posting:
            if doc_id_dest in tf_dict:
                if doc_id_dest not in added:
                    tf_dict[doc_id_dest] += 1
            else:
                added.append(doc_id_dest)
                tf_dict[doc_id_dest] = 1

    # Sort Documents by number unique of tokens in doc
    list_of_dict = sorted([(doc_id, score) for doc_id, score in tf_dict.items()], key=lambda x: x[1], reverse=True)
    # END SOLUTION
    return jsonify(list_of_dict)

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
    # BEGIN SOLUTION
    res = [page_rank[wiki_id] for wiki_id in wiki_ids] 
    # END SOLUTION
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
    # BEGIN SOLUTION
    try:
        res = list(map(lambda x: (page_view[x]), wiki_ids))
    except:
        res = []
        for pageID in wiki_ids:
            try:
                res.append(page_view[pageID])
            except:
                res.append(0) 
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
