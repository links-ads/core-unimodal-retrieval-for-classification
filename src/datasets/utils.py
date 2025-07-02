import pandas as pd
import re
import nltk
import contractions
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours

def preprocess_text(
        txt: str, 
        lst_regex: list = None, 
        punkt: bool = True, 
        lower: bool = True, 
        slang: bool = True, 
        lst_stopwords: bool = None, 
        stemm: bool = False, 
        lemm: bool = True
    ):
    """
        Preprocess a string.
        :parameter
            :param txt: string - name of "text"  containing text
            :param lst_regex: list - list of regex to remove
            :param punkt: bool - if True removes punctuations and characters
            :param lower: bool - if True convert lowercase
            :param slang: bool - if True fix slang into normal words
            :param lst_stopwords: list - list of stopwords to remove
            :param stemm: bool - whether stemming is to be applied
            :param lemm: bool - whether lemmitisation is to be applied
        :return
            cleaned text
    """
    ## Regex (in case, before cleaning)
    if lst_regex is not None: 
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)

    ## Clean 
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang
    txt = contractions.fix(txt) if slang is True else txt
            
    ## Tokenize (convert from string to list)
    lst_txt = txt.split()
                
    ## Stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
                
    ## Lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    ## Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]
            
    ## Back to string
    txt = " ".join(lst_txt)
    return txt