# IR-final-project
## Authors: 
Amit Chen, Omer Hanan.

---
## Summary:
Ben Guriun University, Software and Information Systems Engineering, Information retrieval course - final project.
This project is about creating a search engine to the entire Wikipedia corpus, above 6,300,000 documents (!).
We used all course's theory in order to develop our search engine:
1. Preprocessed the data- 
    - Parsing 
    - Tokenization 
    - remove stopwords
    - Stemming
    - Separated every page to 5 components: title, body, anchor text, page rank, page view.
    - created Inverted Index to Each component.

2. Query Expansion- After trying GloVe (Global Vectors for Word Representation), Word2Vec and WordNet, we decided to use WordNet to perform Query Expansion since this is the fastest approach and it gives pretty good ampiric results. We used 1 best sysnonym and 1 best hypernym (if exists) to every word.</br> For example- searching for tokenized query ["Marijuana"] will expand to ["Marijuana","Cannabis"].

3. Search throught the data. After preprocessing the data we used a simularity measures to determine the simularity between a given query to the engine and a candidate document, using the inverted index. Each document component have it own ranking method, depend on the component content and values.
    - Title: binary ranking method.
    - Body: BM25 simularity measure, with tested K and b values (different from the calculation performed in the `search_body` route which using CosineSimularity measure).
    - Anchor: calculation for anchor rank (different from he calculation performed in the `search_anchor` route, where it is binary).
    - PageRank: calculation for page authority.
    - PageView: calculation for page popularity.

4. Evaluation- The course staff provide us with labled benchmark as test dataset for tesing and evaluation. The evaluation metrics we used are:
    - Precision: number of relevant documents for given query that retrived, divide by number of retrived documents.
    - Recall: number of relevant documents for given query that retrived, divide by all relevant documents.
    - Map@40: # Add explain about Map@40.

 ---
## Indexes and Dictionaries:
We used the following data structures in order to make the claculations faster:
1. DL: <Document id, Length dictionary>
2. DT: <Document id, Title>
3. NF: <Document id, Normalization factor- vector length>

And the following Inverted Inexes to achive fast retrieval:
1. Title Inverted Index (stemmed).
1. Body Inverted Index (stemmed).
1. Anchor text Inverted Index (stemmed).
1. PageRank csv (normalized).
1. PageView dictionary (normalized).

---
## System components (API endpoints):
### Serach / main route:
![](/images/BM25.png)

### Search_body:
![](/images/CosineSim.png)
### Search title:

### Search anchor:

### [POST] PageRank (authority value):

### [POST] PageView (views):
---
## How to run the project:

