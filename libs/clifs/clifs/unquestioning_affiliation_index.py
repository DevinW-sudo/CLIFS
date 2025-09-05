# re-implementation of the Unquestioning Affiliation Index (UAI)
# based on the work of Ashwini Ashokkumar and James W. Pennebaker (2022)
# "Tracking group identity through natural language within groups"
# nUAI modified to remove the z-score calculation

from clifs.uai_codes import *
import nltk
from scipy.stats import zscore
nltk.download('punkt')

def calculate_word_counts(text, word_list):
    words = nltk.word_tokenize(text)
    star = 0
    no_star = 0
    star += count_stars(text, word_list)
    no_star += count_normal(words, word_list)
    count = star + no_star
    return count
    
def count_stars(text_words, word_list):
    # for words in the word list that end
    # with an asterisk, count the number of
    # times any word that starts with the
    # word is found in the text
    count = 0
    stars = [word for word in word_list if word.endswith('*')]
    for star in stars:
        count += len([word for word in text_words if word.startswith(star[:-1])])      
    return count
                                                                    
def count_normal(text_words, word_list):
    count = 0
    no_stars = [word for word in word_list if not word.endswith('*')]
    count = len([word for word in text_words if word in no_stars])
    return count

def calculate_z_scores(df):
    z_score_column1 = "affiliation_z"
    z_score_column2 = "cogproc_z"
    df[z_score_column1] = zscore(df['affiliation'])
    df[z_score_column2] = zscore(df['cogproc'])
    return df

def calculate_uai(df):
    # Calculate the Unquestioning Affiliation Index (UAI)
    df['uai'] = df['affiliation_z'] - df['cogproc_z']
    return df

def calculate_naive_uai(df):
    # Calculate the Unquestioning Affiliation Index (UAI)
    df['uai'] = df['affiliation'] - df['cogproc']
    return df

def calculate_nuai(aff, cp):
    # Calculate the naive UAI
    # for a single text
    nuai = aff - cp
    return nuai

def process_dataframe(df, text_column, naive=False):
    if text_column not in df.columns:
        raise ValueError(f"The column '{text_column}' does not exist in the DataFrame.")

    # Initialize count columns
    df['affiliation'] = df[text_column].apply(lambda x: calculate_word_counts(str(x), affiliation))
    df['cogproc'] = df[text_column].apply(lambda x: calculate_word_counts(str(x), cogproc))

    if naive:
        # Calculate naive uai
        df = calculate_naive_uai(df)

    else:
        # Calculate z-scores for each category
        df = calculate_z_scores(df)
        # calculate uai
        df = calculate_uai(df)

    return df

# Main function for command-line usage
def uai(df, text_column='write', naive=False):
    # Process the DataFrame
    try:
        processed_df = process_dataframe(df, text_column, naive)
    except ValueError as e:
        print(e)
        return

    return processed_df
    
