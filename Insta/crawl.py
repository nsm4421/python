#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import lxml
import urllib
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time
import os

class insta_crawl():
    def __init__(self, search,n_scroll=5):
        self.search = search
        self.n_scroll = n_scroll
        self.search_url = 'https://www.instagram.com/{}'.format(urllib.parse.quote(self.search))
        self.n = 1
        self.url_lst = []
                
    def sleep(self,sleep_time=2):
        time.sleep(sleep_time)

    def collect_url(self):
        self.chrome = webdriver.Chrome()
        self.chrome.get(self.search_url)
        for i in range(int(self.n_scroll)):
            html = self.chrome.page_source
            soup = bs(html, 'lxml')
            insta = soup.select('.v1Nh3.kIKUG')
            for url in insta:
                img_url = url.select_one('.KL4Bh').img['src']
                self.url_lst.append(img_url)
            self.chrome.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.sleep()
        self.chrome.close()
        self.url_lst = list(set(self.url_lst))
        print('{}개의 이미지 URL 확보'.format(len(self.url_lst)))

    def mkdir(self):
        os.mkdir(path = './{}'.format(self.search))

    def save_img(self, url):
        img_name = '{}/{}의 {}번째 이미지.jpg'.format(self.search,self.search, str(self.n))
        urllib.request.urlretrieve(url, img_name)
        print("{}번째 이미지 다운 완료".format(self.n))
        self.n+=1    

