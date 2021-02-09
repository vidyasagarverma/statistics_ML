#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup

URL = 'https://www.livemint.com/Search/Link/Keyword/dixon'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')


# In[2]:


#id="mySearchView"
results = soup.find(id='mySearchView')

print(results.prettify())


# In[3]:


#listing clearfix 
news_elems = results.find_all('div', class_='headlineSec')
print(news_elems)


# In[4]:


for news_elem in news_elems:
    print(news_elem, end='\n'*2)


# In[5]:


for news_elem in news_elems:
    title_elem = news_elem.find('h2', class_='headline')
    
    if None in (title_elem):
        continue
    print(title_elem.text.strip())
print()


# In[19]:


import urllib
import requests
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd
import os


# In[30]:




def find_jobs_from(website, job_title, location, desired_characs, filename="results.csv"):    
    """
    This function extracts all the desired characteristics of all new job postings
    of the title and location specified and returns them in single file.
    The arguments it takes are:
        - Website: to specify which website to search (options: 'mint' or 'indian express')
        - title of the news 
        
        - Filename: to specify the filename and format of the output.
            Default is .xls file called 'results.xls'
    """
    
    if website == 'mint':
        job_soup = load_indeed_jobs_div(job_title, location)
        jobs_list, num_listings = extract_job_information_indeed(job_soup, desired_characs)
    
    if website == 'indianexpress':
        location_of_driver = os.getcwd()
        driver = initiate_driver(location_of_driver, browser='chrome')
        job_soup = make_job_search(job_title, location, driver)
        jobs_list, num_listings = extract_job_information_cwjobs(job_soup, desired_characs)
    
    save_jobs_to_excel(jobs_list, filename)
 
    print('{} new job postings retrieved from {}. Stored in {}.'.format(num_listings, 
                                                                          website, filename))
    


# In[22]:


def load_news_info_div(news_title):
    getVars = {'q' : news_title, 'fromage' : 'last', 'sort' : 'date'}
    url = ('https://www.livemint.com/' + urllib.parse.urlencode(getVars))
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    #id="mySearchView"
    results = soup.find(id='mySearchView')

    return results


# In[23]:


def extract_news_title_indeed(news_elem):
    title_elem = news_elem.find('h2', class_='headline')
    title = title_elem.text.strip()
    return title


# In[24]:



news_elems = results.find_all('div', class_='headlineSec')


# In[25]:


cols = []
extracted_info = []


if 'titles' in desired_characs:
    titles = []
    cols.append('titles')
    for news_elem in news_elems:
        titles.append(extract_news_title_indeed(news_elem))
    extracted_info.append(titles)    


# In[35]:


def find_jobs_from(website, news_title, desired_characs, filename="results.csv"):    
    """
    This function extracts all the desired characteristics of all new job postings
    of the title and location specified and returns them in single file.
    The arguments it takes are:
        - Website: to specify which website to search (options: 'Indeed' or 'CWjobs')
        - Job_title
        - Location
        - Desired_characs: this is a list of the job characteristics of interest,
            from titles, companies, links and date_listed.
        - Filename: to specify the filename and format of the output.
            Default is .xls file called 'results.xls'
    """
    
    if website == 'mint':
        news_soup =load_news_info_div(news_title)
        news_list, num_listings = extract_job_information_indeed(news_soup, desired_characs)
    
    
    
    save_news_to_csv(news_list, filename, delimiter=',', fmt='%s')
 


# In[34]:


def save_news_to_csv(news_list, filename):
    news = pd.DataFrame(news_list)
    news.to_csv(filename)


# In[ ]:




