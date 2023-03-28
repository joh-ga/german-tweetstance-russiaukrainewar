"""
Author: Johanna Garthe
Script to preprocess text with different options
"""

import re
import demoji
import emoji
import wordninja
import pandas as pd
from string import printable

LANGUAGE_MODEL = wordninja.LanguageModel('./customized.txt.gz')
FILENAME = " "

def main():

    """ String of punctuation symbols that should be removed """
    punctuation1 = '!"$%&\'()*+,-.–/:;<=>?[\\]^_`{|}~•@„“”‘’«»⋘⋙♥︎↓®©℗™'
    punctuation2 = '€_'
    
    spellingDict = {
        'mariupol':['marioupol','mariuopol','mariopol','маріуполь','мариуполь'],
        'selenskyj':['selenski','selensky','selenskyy','selenskyi','zelenskiy','zelensky','zelenskyi','zelenskyy','zelinsky'],
        # Etc.
        }
    
    queryhashtags = []

    def remove_links(tweet):
        """ Takes a string and removes web links from it """
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r'bit.ly/\S+', '', tweet)
        tweet = re.sub(r'pic.twitter\S+','', tweet)
        return tweet

    def remove_users(tweet):
        """ Takes a string and removes retweet and @user information """
        tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
        tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
        return tweet

    def map_nondiacritical(in_string: str) -> str:
        """ Takes a string and maps non-diacritical into diacritical marks """
        nondiacriticDict = {'ae':'ä','oe':'ö','ue':'ü'}
        rc = re.compile('|'.join(map(re.escape, nondiacriticDict)))
        def translate(match):
            return nondiacriticDict[match.group(0)]
        out_string = rc.sub(translate, in_string)
        return out_string

    def map_diacritical(in_string: str) -> str:
        """ Takes a string and maps diacritical into non-diacritical marks """
        diacriticDict = {'ä':'ae','ö':'oe','ü':'ue'}
        rc = re.compile('|'.join(map(re.escape, diacriticDict)))
        def translate(match):
            return diacriticDict[match.group(0)]
        out_string = rc.sub(translate, in_string)
        return out_string

    def transform_hashtag(tweet):
        """ Takes a string and removes any hashtags and transforms it from '#WeLoveHashtags' to 'we love hashtags' """
        hashtags = {}
        for word in tweet.split():
            if word.startswith('#'):
                tok = map_diacritical(word)
                transformed = " ".join(LANGUAGE_MODEL.split(tok.lower()))
                transformed_uml = map_nondiacritical(transformed)
                hashtags[word]=transformed_uml            
        for word, replacement in hashtags.items():
            tweet = tweet.replace(word, replacement)
        return tweet
    
    def remove_hashtag(tweet):
        """ Takes a string and removes complete hashtag """
        tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
        return tweet
    
    def remove_query_hashtags(tweet):
        """ Takes a string and removes query hashtags in every tweet """
        tweet = tweet.split()
        resultwords  = [word for word in tweet if word.lower() not in queryhashtags]
        tweet = ' '.join(resultwords)
        return tweet

    def remove_emojis(tweet):
        """ Takes a string and removes all emojis """
        dem = demoji.findall(tweet)
        for item in dem.keys():
            tweet = tweet.replace(item, '')
        return tweet

    def replace_emojis(tweet):
        """ Takes a string and transforms emojis into their German string description, and only keeping the first occurrence in case of emoji duplicates """
        tweet = emoji.demojize(tweet, language='de', version=5.0)
        tweet = tweet.replace(":"," ")
        seen = set()
        tweet = ' '.join(seen.add(i) or i for i in tweet.split() if i not in seen)
        return tweet

    def replace_spelling(tweet):
        """ Replace slang words, abbreviations, and spellings of a word into a specified source word """
        words = tweet.split()
        spellFound={}
        for key, value in spellingDict.items():
            for word in words:
                if word in value:
                    spellFound[word]=key
        for word, replacement in spellFound.items():
            tweet = tweet.replace(word, replacement)
        return tweet

    def clean(tweet):
        """ Main function to clean tweets """
        tweet = remove_users(tweet)                             # Remove @user and RT information
        tweet = remove_links(tweet)                             # Remove links
        tweet = re.sub('&amp;', 'und', tweet)                   # Correct German 'and' token
        tweet = re.sub('[' + punctuation1 + ']+', ' ', tweet)   # Strip punctuation
        #tweet = replace_emojis(tweet)                          # Convert emoji into its textual description
        tweet = remove_emojis(tweet)                            # Remove emojis
        tweet = tweet.lower()                                   # Convert to lowercase
        #tweet = remove_query_hashtags(tweet)                    # Remove query hashtags to exclude obvious clues
        tweet = transform_hashtag(tweet)                        # Transform remaining hashtags
        #tweet = remove_hashtag(tweet)                          # Remove hashtags
        tweet = re.sub(r'(?<!\S)#(\S+)', r'\1', tweet)          # Remove any other hashtag charachters
        tweet = re.sub(r'#(\S+)', r' \1', tweet)                # Remove any other hashtag charachters between tokens as 'word#word' to 'word word'
        tweet = replace_spelling(tweet)                         # Fix spelling variations
        tweet = re.sub('([0-9]+)', '', tweet)                   # Remove numbers
        tweet = re.sub('[' + punctuation2 + ']+', ' ', tweet)   # Strip euro sign character
        tweet = re.sub('\s+', ' ', tweet)                       # Remove double spacing
        tweet = tweet.strip()                                   # Remove leading and trailing whitespaces
        return tweet

    # ----- LOAD FILE ----- #
    df = pd.read_csv('{}.csv'.format(FILENAME))
    tweets = df['text']
    # ----- CLEAN TEXT ----- #
    cleaned = [clean(tweet) for tweet in tweets]
    # ----- SAVE IN NEW COLUMN ----- #
    df.insert(3, "text_cleaned", cleaned)
    df.to_csv('{}_cleaned.csv'.format(FILENAME), index=False, header=True)
    print('********* COMPLETED *********')


if __name__ == "__main__":
    main()
