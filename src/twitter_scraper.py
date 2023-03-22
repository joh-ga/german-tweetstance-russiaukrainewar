"""
Author: Johanna Garthe
Twitter API v2 Full-Archive Search with query containing hashtag groups
"""

import tweepy
import config_twitter
import time
import re
import pandas as pd

# ----- TWITTER AUTHENTICATION ----- #
client = tweepy.Client(bearer_token=config_twitter.BEARER_TOKEN, wait_on_rate_limit=True)

# ----- QUERY-SPECIFIC INFORMATION ----- #
# Hashtag group used for auto-labeling
HASHTAGGROUP = ['#tempolimit','#germanautobahn'] # Etc.
TARGET = "SLI"
stances = ['AGAINST','FAVOR']
LABEL = stances[0]
STARTTIME = '2022-02-24T00:00:00Z'
ENDTIME = '2022-12-31T00:00:00Z' 

def main():
    def format_query(hashtaggroup):
        """ Takes a list of hashtags and converts it into the form (#hashtag1 OR #hashtag2) """
        query_hashtags = '('+' OR '.join(hashtaggroup)+')'
        return query_hashtags

    def twitter_scraper(query,starttime,endtime):
        """ Requests and receives Twitter data in JSON format according to the specified search parameters """
        tweets = []
        for response in tweepy.Paginator(   
            client.search_all_tweets,
            query = query + ' -is:retweet -is:reply lang:de',
            user_fields = ['username','public_metrics','description','location'],
            tweet_fields = ['created_at','geo','public_metrics','text','id','entities','source'],
            expansions = ['author_id','geo.place_id','attachments.media_keys'],
            media_fields = ['media_key','type','url'],
            place_fields = ['country','full_name','geo','id','name'],
            start_time = starttime,
            end_time = endtime,
            max_results = 500,    # Max number per page with system limit of 500
            limit = 300):         # Request page account limit 300 requests per 15 minutes
            time.sleep(1)       # Full-archive 1 request / 1 second limit
            tweets.append(response)
        print('********* API CALL COMPLETED – RESPONSE WITH {} PAGES RECEIVED *********'.format(len(tweets)))
        return tweets

    def json_to_csv(tweets,target,label,hashtaggroup):
        """ Transforms the unordered data structure of the JSON format into a structured form in a CSV file """
        result = []
        user_dict = {}
        attach_dict = {}
        place_dict = {}
        # Loop through each response object
        for response in tweets:
            # Extract all of the users, attachments and places information if exists and put them into a dictionary of dictionaries
            for user in response.includes['users']:
                user_dict[user.id] = {'username': user.username, 
                                    'followers': user.public_metrics['followers_count'],
                                    'tweets': user.public_metrics['tweet_count'],
                                    'description': user.description,
                                    'location': user.location,}
            try:
                for attach in response.includes['media']:
                    attach_dict[attach.media_key] = {
                        'media_key': attach.media_key,
                        'type': attach.type,
                        'url': attach.url,
                        }
                for place in response.includes['places']:
                    place_dict[place.id] = {
                        'country': place.country,
                        'full_name': place.full_name,
                        'geojson': place.geo['bbox'],
                    }
            except:
                pass

            for tweet in response.data:
                # For each tweet, find the author's information
                author_info = user_dict[tweet.author_id]
                # For each tweet, find matching query hashtag
                for hashtag in hashtaggroup:
                    if re.search(hashtag.lower(), tweet.text.lower()):
                        query_hashtag = hashtag
                    else:
                        query_hashtag = None
                # For each tweet, find all hashtags, urls, attachments and geodata if exist by the respective id
                try:
                    if tweet.geo is not None:
                        place_info = place_dict[tweet.geo['place_id']]
                except:
                    pass
                tags = []
                urls = []
                attachments = []
                try:
                    [tags.append(', '.join(["#"+str(h['tag']) for h in tweet.entities['hashtags']]))]
                    [urls.append(u['url']) for u in tweet.entities['urls']]
                    if tweet.attachments is not None:
                        for val in tweet.attachments.values():
                            for v in val:
                                attachments.append(attach_dict[v])
                except:
                    pass

                # Put all information in a single dictionary per tweet
                result.append({'text_id': tweet.id,
                            'text': tweet.text.replace('\n', ' ').replace('\r', ' '),
                            'target': target,
                            'stance_label': label,
                            'query': query_hashtag,
                            'text_hashtags': [tags if tags else None],
                            'text_urls': [urls if urls else None],
                            'attachments': [attachments if attachments else None],
                            'created_at': tweet.created_at,
                            'country': [place_info['country'] if tweet.geo else None],
                            'location': [place_info['full_name'] if tweet.geo else None],
                            'place_geojson': [place_info['geojson'] if tweet.geo else None],
                            'source': tweet.source,
                            'retweets': tweet.public_metrics['retweet_count'],
                            'replies': tweet.public_metrics['reply_count'],
                            'likes': tweet.public_metrics['like_count'],
                            'quote_count': tweet.public_metrics['quote_count'],
                            'author_id': tweet.author_id, 
                            'username': author_info['username'],
                            'author_followers': author_info['followers'],
                            'author_tweets': author_info['tweets'],
                            'author_description': [author_info['description'].replace('\n', ' ').replace('\r', ' ') if author_info['description'] is not None else None],
                            'author_location': [author_info['location'].replace('\n', ' ').replace('\r', ' ').strip() if author_info['location'] is not None else None],
                            })

        df = pd.DataFrame(result)
        df.to_csv('{}_{}.csv'.format(target.lower(), label.lower()), index=False, header=True)
        print('********* EXTRACTION OF JSON INFORMATION INTO CSV FILE COMPLETED ********* \n TOTAL AMOUNT OF DATA: {}'.format(len(df)))

    # ----- MAIN ----- #
    query = format_query(HASHTAGGROUP)
    tweets = twitter_scraper(query,STARTTIME,ENDTIME)
    json_to_csv(tweets,TARGET,LABEL,HASHTAGGROUP)


if __name__ == "__main__":
    main()