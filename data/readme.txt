This directory contains, datasets for sentiment analysis (imdb, amazon, yelp, merged) and for qa (qa_1000, qa_5500)

=======
Format:
=======
sentence \t class \n

=======
Details:
=======
For sentiment analysis, class is either 1 (for positive) or 0 (for negative)
For QA, class ranges from 0 to 5
    "DESC": 0,
    "ENTY": 1,
    "ABBR": 2,
    "HUM": 3,
    "NUM": 4,
    "LOC": 5

For the full datasets look:
imdb: Maas et. al., 2011 'Learning word vectors for sentiment analysis'
amazon: McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating dimensions with review text'
yelp: Yelp dataset challenge http://www.yelp.com/dataset_challenge
