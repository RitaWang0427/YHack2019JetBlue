import nltk
import twitter
import matplotlib.pyplot as plt
# 1. initialize api instance
twitter_api = twitter.Api(consumer_key='wLCSHVI4ajW48JeOyZ1Gyi6bz',
                        consumer_secret='zVTC83LEfzBX3fU8Db1P7hYUSW4ByuzYU8c4KmRuLNsATBbAA7',
                        access_token_key='1187921319482089473-yfrZqZmW2DNEcjbQvxk7XtHvXSE8EA',
                        access_token_secret='DIN7lDEIOCKiStBSCS4To1MeWiH5RJs35wuIv9i4irkR8')

# test authentication in JSON format
print(twitter_api.VerifyCredentials())

# *** 180 tweets per 15 mins

# 2. Creating the function to build the Test set
def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count=100)

        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)

        return [{"text": status.text, "label": None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None

search_term = input("Enter a search keyword:")
testDataSet = buildTestSet(search_term)

print(testDataSet[0:4]) # print out five tweets that contain our search keyword on the Terminal of your IDE

# 3. Preparing The Training Set

# max_number_of_requests = 180
# time_window = 15 minutes = 900 seconds
# Therefore, the process should follow:
# Repeat until end-of-file: {
#     180 requests -> (900/180) sec wait
# }

def buildTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time

    corpus = []

    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id": row[2], "label": row[1], "topic": row[0]})

    rate_limit = 180
    sleep_time = 900 / 180

    trainingDataSet = []

    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched" + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            time.sleep(sleep_time)
        except:
            continue
    # now we write them to the empty CSV file
    with open(tweetDataFile, 'w') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet

corpusFile = "/Users/oumizuyou/PycharmProjects/YHack2019test/corpus1.csv"
tweetDataFile = "/Users/oumizuyou/PycharmProjects/YHack2019test/tweetDataFile.csv"

trainingData = buildTrainingSet(corpusFile, tweetDataFile)



# pre-processing
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets

    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]

tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)

# Naive Bayes Classifier

#  Building the vocabulary
def buildVocabulary(preprocessedTrainingData):
    all_words = []
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features

# Building our feature vector
word_features = buildVocabulary(preprocessedTrainingSet)
delay = ['delay','wait','long','speed','sit','cancel']
baggage = ['broken','lose','baggage','miss','luggage','damage']

# Matching tweets against our vocabulary
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


trainingFeatures = nltk.classify.apply_features(extract_features, preprocessedTrainingSet)

# Training the classifier
NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

# Testing The Model:
NBResultLabels = []
NBResultTopics=[]
for tweet in preprocessedTestSet:
    classifier = NBayesClassifier.classify(extract_features(tweet[0]))
    NBResultLabels.append(classifier)
    if classifier=='negative':
        if set(tweet[0]).intersection(set(delay)) != set():
            NBResultTopics.append('delay')
        elif set(tweet[0]).intersection(set(baggage)) != set():
            NBResultTopics.append('baggage')
        else: NBResultTopics.append('customer services')

# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
print("Reasons for Negative Sentiment: ")
x = 100*NBResultTopics.count('delay')/len(NBResultTopics)
y = 100*NBResultTopics.count('baggage')/len(NBResultTopics)
z = 100*NBResultTopics.count('customer services')/len(NBResultTopics)
print("Percentage of Airline Time Issues = " + str(x) + "%")
print("Percentage of Customers' Baggage Issues = " + str(y) + "%")
print("Percentage of Customer Services Issues = " + str(100*NBResultTopics.count('customer services')/len(NBResultTopics)) + "%")
labels = ["Airline Time","Customers' Baggage","Customer Services"]
size = [x,y,z]
colors = ['#00CCCC','#006666','#009999']
plt.pie(size,labels=labels,colors = colors,autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.show()
