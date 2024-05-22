from bs4 import BeautifulSoup
from bs4.element import Comment
from transformers import AutoModelForCausalLM, AutoTokenizer 
from urllib.parse import urlparse
import requests, json, transformers, urllib.parse, posixpath, os, html2text 



 # request headers
headers = {
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36', # change to your local system or browser
    'Accept-Language' : 'en-GB,en;q=0.5', 
    'DNT': '1', 
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'cache-control': 'max-age=0', 
    'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Brave";v="122"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"', #change to local os 
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'sec-gpc': '1',
}


def get_webpage():

    url = input("Paste Website Link here:")

    params = False # params dont exist. data-page unknown 

    r = requests.get(url, headers=headers) # no params because whole page is being searched with no specific filter 

    if r.status_code == 200: 
        with open('index.html', 'w', encoding='utf-8') as file:
            file.write(r.text)
        print("Data Succesfully Saved in HTML file! Processing further tokenization .......")

        with open('index.html', 'r', encoding='utf-8') as file:
            file_content = file.read()

        return file_content, url 

    else: 
        print("Unable to retrieve data from website :( Please try a different link!")
        return None, None 


def get_text(html_content, base_url): 

    page_info = {}

    

    soup = BeautifulSoup(html_content, 'html.parser')

    title = soup.find("title")

    if title:
        page_info['search'] = title.string

    domain = urlparse(base_url).netloc

    page_info['website-name'] = domain

    texts = soup.find_all(string=True)

    visible_texts = filter(validate_text, texts)  

    one_string = u" ".join(t.strip() for t in visible_texts)

    page_info['page_content'] = one_string

   
    return page_info

   

    
    
def validate_text(input): 

    black_list = ['style', 'head', 'script', 'meta', 'br', 'input', 'button']

    if input.parent.name in black_list: 
        return False
    
    if isinstance(input, Comment): 
        return False 
    
    return True


def get_visual(html_content, base_url): 

    soup = BeautifulSoup(html_content, 'html.parser') 
    soup.prettify()



    """ Gather Images -> Split into collection of images"""
    images = soup.find_all("img")

    if not os.path.exists('./images'): 
        os.mkdir('./images')


    for source in images:  # locate the src of the image 
        

        img = source['src']
        #image_sources.append(img)
        url_img_path = posixpath.join(img)

        absolute_url = urllib.parse.urljoin(base_url, url_img_path)

        get_image = requests.get(absolute_url, headers=headers)

        if get_image.status_code == 200: 

            img_name = os.path.basename(urllib.parse.urlparse(absolute_url).path) 

            image_path = os.path.join('images', img_name)

            with open(image_path, 'wb') as file: 
                file.write(get_image.content)
            #print("Image has been saved succesfully!")
        
        else: 
            print(f"Failed to download image from the original url: {absolute_url}")
        


    
def tokenize(sorted_info):

    checkpoint = "HuggingFaceH4/zephyr-7b-beta"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    tk_string = sorted_info

    first_val = next(iter(tk_string.values()))

    second_val = list(tk_string.values())[1]

    last_val = list(tk_string.values())[-1]

    messages = [

        {
            "role": "system",
            "content": f"you are looking at the extracted data of a website. The website is {second_val} with the bio of {first_val}."
        
        },

        {
            "role": "user",
            "content": f"The contents of this website are {last_val}"
        }

    ]

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    decode = tokenizer.decode(tokenized_chat[0])

    with open('trainingdatatokenized.txt', 'w', encoding='utf-8') as file:
            file.write(decode)
    print("Data Succesfully Saved in .TXT file! Tokenization Complete!")

    



   
    


data_content, url_content  = get_webpage()

if data_content: 
    get_visual(data_content, url_content)

else: 
    print("No Data To Parse!")



get_text(data_content, url_content)

tokenize(get_text(data_content, url_content))







    

