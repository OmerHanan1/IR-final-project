import numpy as np
import bz2
from functools import partial
from collections import Counter
import pickle
from itertools import islice
from xml.etree import ElementTree
import codecs
import csv
import time
import os
import re
from pathlib import Path

# Paths
# Using user page views (as opposed to spiders and automated traffic) for the 
# month of August 2021
pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
p = Path(pv_path) 
pv_name = p.name
pv_temp = f'{p.stem}-4dedup.txt'
pv_clean = f'{p.stem}.pkl'
# Download the file (2.3GB) 
!wget -N $pv_path
# Filter for English pages, and keep just two fields: article ID (3) and monthly 
# total number of page views (5). Then, remove lines with article id or page 
# view values that are not a sequence of digits.
!bzcat $pv_name | grep "^en\.wikipedia" | cut -d' ' -f3,5 | grep -P "^\d+\s\d+$" > $pv_temp
# Create a Counter (dictionary) that sums up the pages views for the same 
# article, resulting in a mapping from article id to total page views.
wid2pv = Counter()
with open(pv_temp, 'rt') as f:
  for line in f:
    parts = line.split(' ')
    wid2pv.update({int(parts[0]): int(parts[1])})
# write out the counter as binary file (pickle it)
with open(pv_clean, 'wb') as f:
  pickle.dump(wid2pv, f)
  
# read in the counter
with open(pv_clean, 'rb') as f:
  wid2pv = pickle.loads(f.read())
  