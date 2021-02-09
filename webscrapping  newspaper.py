#!/usr/bin/env python
# coding: utf-8




import requests
from bs4 import BeautifulSoup

URL = 'https://www.livemint.com/Search/Link/Keyword/dixon'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')


#id="mySearchView"
results = soup.find(id='mySearchView')

print(results.prettify())



#listing clearfix 
news_elems = results.find_all('div', class_='headlineSec')
print(news_elems)



for news_elem in news_elems:
    print(news_elem, end='\n'*2)


# In[5]:


for news_elem in news_elems:
    title_elem = news_elem.find('h2', class_='headline')
    
    if None in (title_elem):
        continue
    print(title_elem.text.strip())
print()


