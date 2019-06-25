# -*- coding: utf-8 -*-
from selenium import webdriver
import time
import pandas as pd
import re
from selenium.webdriver.common.keys import Keys
import numpy as np
import csv
class SeleniumClient(object):
    def __init__(self):
        #Initialization method. 
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-setuid-sandbox')

        # you need to provide the path of chromdriver in your system
        self.browser = webdriver.Chrome('C:\Windows\chromedriver_win32\chromedriver', options=self.chrome_options)

        self.base_url = 'https://twitter.com/search?q='
    
    def remove_pattern(self,text, pattern_regex):
        r = re.findall(pattern_regex, text)
        for i in r:
            text = re.sub(i, '', text)
        return text

    def get_tweets(self, query):
        ''' 
        Function to fetch tweets. 
        '''
        try: 
            self.browser.get(self.base_url+query)
            time.sleep(2)

            body = self.browser.find_element_by_tag_name('body')

            for _ in range(200):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)
            with open('getdata.csv','a')as csv_file:
                writer = csv.writer(csv_file)
                timeline = self.browser.find_element_by_id('timeline')
                tweet_nodes = timeline.find_elements_by_css_selector('.tweet-text')
                for tweet_node in tweet_nodes:
                    tweet = self.remove_pattern(tweet_node.text, "@|#[\w]*")
                    tweet = self.remove_pattern(tweet, "http|https|com|//|/|pic.twitter|:")
                    writer.writerow([tweet,'1'])
            csv_file.close()
                
        except: 
            print("Selenium - An error occured while fetching tweets.")