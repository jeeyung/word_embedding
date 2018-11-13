import tweepy
import os
import multiprocessing as mp

class TwitterCrawler:
    def __init__(self):
        self.CONSUMER_KEY = "euEIj2HoqGtwEH3ZmT3s8LRzM"
        self.CONSUMER_SECRET = "3Gign2g0DEV5RhaIfZJMkForTSThRhDMuTePb1cnMWJK2L2WNg"
        self.ACCESS_TOKEN = "108031820-8fmQ91pKAzRH6tH27cv2cdN3GiXksUjTu0pDku3H"
        self.ACCESS_TOKEN_SECRET = "GxfzEkGh5KJ53bAlsHJiwR4f1R2CG1Vytn03jHN9Ifc0k"
        self.SAVE_FILE = "/twitter.txt"
        self.api = self.generateAPI()

    def generateAPI(self):
        auth = tweepy.OAuthHandler(self.CONSUMER_KEY, self.CONSUMER_SECRET)
        # access 토큰 요청
        auth.set_access_token(self.ACCESS_TOKEN, self.ACCESS_TOKEN_SECRET)
        # twitter API 생성
        api = tweepy.API(auth, wait_on_rate_limit=True)
        print("API generated")
        return api

    def requestCrawl(self, keyword, location):
        # 검색기준(대한민국 중심) 좌표, 반지름
        # OR 로 검색어 묶어줌, 검색어 5개(반드시 OR 대문자로)
        wfile = open(os.getcwd() + self.SAVE_FILE, mode='w')     # 쓰기 모드
        # twitter 검색 cursor 선언
        cursor = tweepy.Cursor(self.api.search,
                               q=keyword,
                               lang = 'en',
                               tweet_mode = 'extended',
                               since='2015-01-01',  # 2015-01-01 이후에 작성된 트윗들로 가져옴
                               count=100,           # 페이지당 반환할 트위터 수 최대 100
                               geocode=location,    # 검색 반경 조건
                               include_entities=True)
        print("Crawling start")
        for i, tweet in enumerate(cursor.items()):
            if "retweeted_status" in dir(tweet):
                # text = tweet.retweeted_status.full_text
                continue
            else:
                try:
                    text = tweet.full_text
                except AttributeError:
                    text = tweet.text
            # print("{}: {}".format(i, text))
            wfile.write(text + '\n')
            if i > 5000:
                break
        wfile.close()


    def streamCrawl(self, keywords, queue):
        myStreamListener = MyStreamListener(queue)
        myStream = tweepy.Stream(auth=self.api.auth, listener=myStreamListener)
        myStream.filter(track=keywords)


class MyStreamListener(tweepy.StreamListener):
    def __init__(self, queue):
        super(MyStreamListener, self).__init__()
        self.queue = queue

    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            try:
                tweet = status.retweeted_status.extended_tweet["full_text"]
            except:
                tweet = status.retweeted_status.text
        else:
            try:
                tweet = status.extended_tweet["full_text"]
            except AttributeError:
                tweet = status.text
        # print(tweet)
        self.queue.put(tweet)

    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False

if __name__ == "__main__":
    crawler = TwitterCrawler()
    # crawler.requestCrawl("president", None)
    # crawler.requestCrawl("president OR trump", "%s,%s,%s" % ("38.53", "-77.03", "1000km"))
    queue = mp.Queue()
    crawler.streamCrawl(["trump"], queue)