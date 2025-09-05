# lite re-implementation of the violence risk index. Removed categories unnecessary for
# our fusion analysis (also removed harmful categories; e.g. ones including racial slurs)
# implemented in Python instead of R
# based on the work of Julia Ebner, Christopher Kavanagh, and Harvey Whitehouse (2023)
# "Measuring socio-psychological drivers of extreme violence in online terrorist manifestos: an alternative linguistic risk assessment model"

import re
from clifs.vri_codes_lite_clifs import *
import nltk

nltk.download('punkt')

# Function to process each text
def process_text(text):
    new_dict = {}
    # split into sentences
    sentences = nltk.sent_tokenize(text)
    
    for category in categorieslist:
        if '%' in category:  # If the category contains '%', handle as a ratio between two sets of keywords
            part1 = globals()[category.replace('%', '1')]
            part2 = globals()[category.replace('%', '2')]
            sentences_filtered = [s for s in sentences if re.search('|'.join(part1), s, re.IGNORECASE)]
            sentences_filtered2 = [s for s in sentences if re.search('|'.join(part2), s, re.IGNORECASE)]
            if len(sentences_filtered2) == 0:
                temp = 0
            else:
                temp = len(sentences_filtered) / len(sentences_filtered2)

        else:  # Handle a simple category without special characters
            sentences_filtered = [s for s in sentences if re.search('|'.join(globals()[category]), s, re.IGNORECASE)]
            if len(sentences) == 0:
                temp = 0
            else:
                temp = len(sentences_filtered) / len(sentences) * 100

        # Store the result in the DataFrame for the current category
        # category without special characters
        category_name = category
        if '%' in category:
            category_name = category_name.replace('%', '')
        
        new_dict[category_name] = 0 if temp is None else temp

    return new_dict
