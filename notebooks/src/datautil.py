import os, re
from zipfile import ZipFile



def get_articles(dataset_file, search_term=None):    
    articles = []
    
    input_zip = ZipFile(dataset_file)
    for file_name in input_zip.namelist():
        with input_zip.open(file_name) as file:
            for line in file:
                line = line.strip()
                article = "{}".format(line.decode("utf-8"))
                
                if search_term is not None and re.search(r"{}".format(search_term), article, flags=re.IGNORECASE) is None:
                    continue
                
                articles.append(article)

    return articles



def get_random_article(dataset_file):
    articles = []
    
    input_zip = ZipFile(dataset_file)
    for file_name in input_zip.namelist():
        lines = input_zip.open(file_name).read().splitlines()
        return random.choice(lines).decode("utf-8")

    
    