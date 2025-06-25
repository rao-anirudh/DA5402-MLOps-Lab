from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
from bson.binary import Binary


def scrape_google_news(config):

    """
    Uses a configuration given as a Python dictionary to scrape headlines, thumbnails, and other metadata
    from Google News and return it as a list of dictionaries
    :param config (dict) - Python dictionary containing the configuration
    :returns items (list) - Python list containing the scraped data in the form of dictionaries

    """

    driver = webdriver.Chrome()

    driver.get(config["url"])
    time.sleep(config["load_time"])
    driver.find_element(By.LINK_TEXT, config["top_stories_string"]).click()
    time.sleep(config["load_time"])
    previous_height = driver.execute_script('return document.body.scrollHeight')

    items = []

    while True:

        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(config["load_time"])
        new_height = driver.execute_script('return document.body.scrollHeight')

        articles = driver.find_elements(By.CSS_SELECTOR, config["article_class"])
        headlines = driver.find_elements(By.CSS_SELECTOR, config["headline_class"])
        thumbnails = driver.find_elements(By.CSS_SELECTOR, config["thumbnail_class"])
        newspapers = driver.find_elements(By.CSS_SELECTOR, config["newspaper_class"])
        dates = driver.find_elements(By.CSS_SELECTOR, config["date_class"])

        for i in range(len(articles)):
            if "FIGURE" not in [child.get_property("tagName") for child in articles[i].get_property("childNodes")]:
                thumbnails.insert(i, None)
                continue
            headline = headlines[i].get_property("text")
            thumbnail_url = thumbnails[i].get_property("src")
            response = requests.get(thumbnail_url)
            thumbnail = Binary(response.content)
            newspaper = newspapers[i].get_property("textContent").split("More")[0]
            article_date = dates[i].get_property("dateTime").split("T")[0]
            url = headlines[i].get_property("href")
            scrape_time = time.ctime()
            item = {"headline": headline,
                    "thumbnail": thumbnail,
                    "newspaper": newspaper,
                    "article_date": article_date,
                    "url": url,
                    "scrape_time": scrape_time}
            items.append(item)

        if new_height == previous_height:
            break
        previous_height = new_height

    driver.quit()

    return items
