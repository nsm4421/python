#!/usr/bin/env python
# coding: utf-8

# In[4]:


import crawl
insta_id = input('Insta Id : ')
n_scroll = input('How many times to scroll : ')

crawler = crawl.insta_crawl(insta_id, n_scroll)
crawler.collect_url()
crawler.mkdir()
[crawler.save_img(url) for url in crawler.url_lst]


# In[ ]:




