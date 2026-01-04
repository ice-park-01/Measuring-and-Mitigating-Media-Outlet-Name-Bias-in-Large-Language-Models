import os
import sys
import json
import torch
import xml.etree.ElementTree as ET
import random

from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer  
from dataclasses import dataclass

from .prompt import BASE_PROMPT_ALLSIDES, COT_PROMPT_ALLSIDES, BASE_PROMPT_SUMMARIZATION, BASE_PROMPT_ALLSIDES_ORDER, BASE_PROMPT_SUMMARIZATION_LEFT, BASE_PROMPT_SUMMARIZATION_CENTER, BASE_PROMPT_SUMMARIZATION_RIGHT


@dataclass
class Question:
    news_name: str
    title: str
    date: str
    content: str
    label: str
    id: str
    def get_natural_prompt(self, news_name: str):
        prompt = BASE_PROMPT_ALLSIDES
        if news_name == "itself":
            prompt += "\n\nThis article is from " + self.news_name + ".\n\n"
        elif news_name == "none":
            prompt += "\n\n"
        else:
            prompt += "\n\nThis article is from " + news_name + ".\n\n"
        prompt += "###Content starts.\n" + str(self.title) + "\n" + str(self.date) + "\n" + str(self.content) + "\n\n"
        prompt += "###Content ended. The bias is: "
        
        return prompt
    
    def get_natural_prompt_for_order(self, news_name: str):
        order_list = ["lrc", "rlc", "rcl", "crl", "clr"]
        order = random.choice(order_list)
        if order == "lrc":
            bias_1 = "Left"
            bias_2 = "Right"
            bias_3 = "Center"
        elif order == "rlc":
            bias_1 = "Right"
            bias_2 = "Left"
            bias_3 = "Center"
        elif order == "rcl":
            bias_1 = "Right"
            bias_2 = "Center"
            bias_3 = "Left"
        elif order == "crl":
            bias_1 = "Center"
            bias_2 = "Right"
            bias_3 = "Left"
        elif order == "clr":
            bias_1 = "Center"
            bias_2 = "Left"
            bias_3 = "Right"
            
        prompt = BASE_PROMPT_ALLSIDES_ORDER.format(bias_1=bias_1, bias_2=bias_2, bias_3=bias_3)
        
        if news_name == "itself":
            prompt += "\n\nThis article is from " + self.news_name + ".\n\n"
        elif news_name == "none":
            prompt += "\n\n"
        else:
            prompt += "\n\nThis article is from " + news_name + ".\n\n"
        prompt += "###Content starts.\n" + str(self.title) + "\n" + str(self.date) + "\n" + str(self.content) + "\n\n"
        prompt += "###Content ended. The bias is: "
        
        return prompt, order

    def get_natural_prompt_optimization(self, prompt: str, news_name: str):
        prompt = prompt
        if news_name == "itself":
            prompt += "\n\nThis article is from " + self.news_name + ".\n\n"
        elif news_name == "none":
            prompt += "\n\n"
        else:
            prompt += "\n\nThis article is from " + news_name + ".\n\n"
        prompt += "###Content starts.\n" + str(self.title) + "\n" + str(self.date) + "\n" + str(self.content) + "\n\n"
        prompt += "###Content ended. The bias is: "
        
        return prompt
    
    def get_reasoning_prompt(self, news_name: str):
        prompt = BASE_PROMPT_ALLSIDES
        if news_name == "itself":
            prompt += "\n\nThis article is from " + self.news_name + ".\n\n"
        elif news_name == "none":
            prompt += "\n\n"
        else:
            prompt += "\n\nThis article is from " + news_name + ".\n\n"
        prompt += "###Content starts.\n" + str(self.title) + "\n" + str(self.date) + "\n" + str(self.content) + "\n\n"
        prompt += "###Content ended. Please show your choice in the answer field with only the choice letter in json format, e.g., {\"answer\": \"C\"}. "
        
        return prompt
    
    # def get_summarization_prompt(self, news_name: str, summary_length: int):
    #     prompt = BASE_PROMPT_SUMMARIZATION.format(article=self.content, summary_length=summary_length)
    #     if news_name == "itself":
    #         prompt += "\n\nThis article is from " + self.news_name + ".\n\n"
    #     elif news_name == "none":
    #         prompt += "\n\n"
    #     else:
    #         prompt += "\n\nThis article is from " + news_name + ".\n\n"
            
    #     prompt += "Summary: "
    #     return prompt
    
    def get_summarization_prompt(self, news_name: str, summary_length: int):
        prompt = BASE_PROMPT_SUMMARIZATION.format(article=self.content, summary_length=summary_length)
        if news_name == "itself":
            prompt += "\n\nYour summarization will be posted on " + self.news_name + ".\n\n"
        elif news_name == "none":
            prompt += "\n\n"
        else:
            prompt += "\n\nYour summarization will be posted on " + news_name + ".\n\n"
            
        prompt += "Summary: "
        return prompt
    
    def get_summarization_prompt_reference(self, reference_class: str, summary_length: int):
        if reference_class == "left":
            prompt = BASE_PROMPT_SUMMARIZATION_LEFT.format(article=self.content, summary_length=summary_length)
        elif reference_class == "center":
            prompt = BASE_PROMPT_SUMMARIZATION_CENTER.format(article=self.content, summary_length=summary_length)
        elif reference_class == "right":
            prompt = BASE_PROMPT_SUMMARIZATION_RIGHT.format(article=self.content, summary_length=summary_length)
        else:
            raise ValueError(f"Invalid reference class: {reference_class}")
        
        prompt += "Summary: "
        return prompt
    
    def get_cot_prompt(self, news_name: str):
        prompt = COT_PROMPT_ALLSIDES
        if news_name == "itself":
            prompt += "\n\nThis article is from " + self.news_name + ".\n\n"
        elif news_name == "none":
            prompt += "\n\n"
        else:
            prompt += "\n\nThis article is from " + news_name + ".\n\n"
        prompt += "###Content starts.\n" + str(self.title) + "\n" + str(self.date) + "\n" + str(self.content) + "\n\n"
        prompt += "###Content ended. Let's think step by step.\n"
        
        return prompt
    

class CustomDatasetAllsides(Dataset):
    def __init__(self, dataset_path: str, model_name: str, closed_source: bool):
        self.raw_data = []
        self.structured_data = []
        self.prompted_data = []
        self.tokenized_data = []
        self.labels = []
        
        if not closed_source:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.tokenizer = None

        # self.load_dataset_allsides(dataset_path)
        self.load_dataset_custom(dataset_path)
        self.preprocess_data()
        self.construct_prompt()
        
    def __len__(self):
        return len(self.prompted_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.prompted_data[idx]
        return sample
    
    def load_dataset_allsides(self, dataset_path):
        files = os.listdir(dataset_path)
        for file in tqdm(files, desc="Loading dataset"):
            with open(os.path.join(dataset_path, file), "r") as f:
                data = json.load(f)
                self.raw_data.append(data)
                
    def load_dataset_custom(self, dataset_path):
        with open(dataset_path, "r") as f:
            self.raw_data = json.load(f)
    
    def preprocess_data(self):
        for data in tqdm(self.raw_data, desc="Preprocessing data"):
            topic = data['topic']
            source = data['source']
            bias_text = data['bias_text']
            title = data['title']
            date = data['date']
            content = data['content']
            id = data['ID']
            
            self.structured_data.append({
                'topic': topic,
                'source': source,
                'bias_text': bias_text,
                'title': title,
                'date': date,
                'content': content,
                'id': id
            })
            
    def construct_prompt(self):
        for data in tqdm(self.structured_data, desc="Constructing prompt"):
            self.prompted_data.append(Question(data['source'], data['title'], data['date'], data['content'], data['bias_text'], data['id']))
            
    def print_all_bias_texts(self):
        bias_texts = set()
        for data in self.structured_data:
            bias_texts.add(data['bias_text'])
        print("Possible bias_texts:", bias_texts)
        
    def print_all_news_names(self):
        news_names = set()
        for data in self.structured_data:
            news_names.add(data['source'])
        print("Possible news_names:", news_names)
        
        
class CustomDatasetHyperpartisan(Dataset):
    def __init__(self, dataset_path: str, model_name: str, closed_source: bool):
        self.raw_data = []
        self.structured_data = []
        self.prompted_data = []
        self.tokenized_data = []
        self.labels = []
        
        if not closed_source:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.tokenizer = None

        self.load_dataset_hyperpartisan(dataset_path)
        self.preprocess_data()
        self.construct_prompt()
        
    def __len__(self):
        return len(self.prompted_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.prompted_data[idx]
        return sample
    
    def load_dataset_hyperpartisan(self, dataset_path):
        def load_xml_articles(path):
            tree = ET.parse(path)
            root = tree.getroot()

            articles = []
            for article in root.findall('article'):
                article_id = article.get('id')
                published_at = article.get('published-at')
                title = article.get('title')
                content = ''.join(article.itertext()).strip()   
                
                articles.append({
                    'id': article_id,
                    'published_at': published_at,
                    'title': title,
                    'content': content
                })
            return articles
    
        def load_xml_labels(path):
            # XML 파일 로드
            tree = ET.parse(path)  # 파일 경로에 맞게 수정
            root = tree.getroot()

            articles = []
            for article in root.findall('article'):
                articles.append({
                    'id': article.get('id'),
                    'url': article.get('url'),
                    'hyperpartisan': article.get('hyperpartisan'),
                    'labeled_by': article.get('labeled-by')
                })
            return articles
    
        articles_training_path = os.path.join(dataset_path, "articles-training-byarticle-20181122.xml")
        articles_test_path = os.path.join(dataset_path, "articles-test-byarticle-20181207.xml")
        labels_training_path = os.path.join(dataset_path, "ground-truth-training-byarticle-20181122.xml")
        labels_test_path = os.path.join(dataset_path, "ground-truth-test-byarticle-20181207.xml")

        self.articles_training = load_xml_articles(articles_training_path)
        self.articles_test = load_xml_articles(articles_test_path)
        self.labels_training = load_xml_labels(labels_training_path)
        self.labels_test = load_xml_labels(labels_test_path)
        
    def preprocess_data(self):
        for article in tqdm(self.articles_training, desc="Preprocessing training data"):
            id = article['id']
            published_at = article['published_at']
            title = article['title']
            content = article['content']
            
            corresponding_label = next((label for label in self.labels_training if label['id'] == id), None)
            if corresponding_label is None:
                print(f"No corresponding label found for article {id}")
                continue
            
            url = corresponding_label['url']
            news_name = url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
            hyperpartisan = corresponding_label['hyperpartisan']
            
            self.structured_data.append({
                'topic': "hyperpartisan",
                'source': news_name,
                'bias_text': "Hyperpartisan" if hyperpartisan == "true" else "Non-hyperpartisan",
                'title': title,
                'date': published_at,
                'content': content,
                'id': id
            })
        
        for article in tqdm(self.articles_test, desc="Preprocessing test data"):
            id = article['id']
            published_at = article['published_at']
            title = article['title']
            content = article['content']
            
            corresponding_label = next((label for label in self.labels_test if label['id'] == id), None)
            if corresponding_label is None:
                print(f"No corresponding label found for article {id}")
                continue
            
            url = corresponding_label['url']    
            news_name = url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
            hyperpartisan = corresponding_label['hyperpartisan']
            
            if title is None:
                print(f"No title found for article {id}")
                title = ""
                
            if content is None:
                print(f"No content found for article {id}")
                continue
            
            if published_at is None:
                print(f"No date found for article {id}")
                published_at = ""
            
            self.structured_data.append({
                'topic': "hyperpartisan",
                'source': news_name,
                'bias_text': "Hyperpartisan" if hyperpartisan == "true" else "Non-hyperpartisan",
                'title': title,
                'date': published_at,
                'content': content,
                'id': id
            })
                        
    def construct_prompt(self):
        for data in tqdm(self.structured_data, desc="Constructing prompt"):
            self.prompted_data.append(Question(data['source'], data['title'], data['date'], data['content'], data['bias_text'], data['id']))
                    
    def print_all_bias_texts(self):
        bias_texts = set()
        for data in self.structured_data:
            bias_texts.add(data['bias_text'])
        print("Possible bias_texts:", bias_texts)
        
    def print_all_news_names(self):
        news_names = set()
        for data in self.structured_data:
            news_names.add(data['source'])
        print("Possible news_names:", news_names)
    
if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        hf_token = "Your HF Token"
    os.environ["HF_TOKEN"] = hf_token
    # dataset = CustomDatasetAllsides("../../data/allsides/Article-Bias-Prediction/data/jsons", "meta-llama/Llama-3.3-70B-Instruct")
    # dataset.print_all_news_names()
    
    # newses = ['Socialist Worker', 'Austin American-Statesman', 'Glamour', 'Project Syndicate', 'Bustle', 'The Massachusetts Daily Collegian', 'Civil.Ge', 'The Flip Side', 'Toronto Star', 'Nieman Lab', 'Daily Targum', 'Carrie Lukas', 'News Wise', 'Medium', 'GQ.com', 'Honolulu Civil Beat', 'New York Daily News', 'Lisa Gable', 'Guest Writer', 'New York Times Editorial Board', 'Harvard Business School', 'Vox', 'Volante', 'Ross Douthat', 'Sierra Sun', 'Brock Press', 'Dennis Prager', 'The Conversation', 'Rahm Emanuel', 'KALW.org', 'Al Cardenas', 'Reason', 'The Denver Post', 'Pacific Standard', 'The Times Higher Education', 'Mediaite', 'University of Delaware Review', 'Tom Cole', 'IVN', 'CNET', 'Washington Monthly', 'Wall Street Journal - Editorial', 'WND.com', 'Ben Stein', 'The Imaginative Conservative', 'Newsmax - News', 'ABC News (Online)', 'Peter Thiel', 'Ken Burns', 'The Daily Signal', 'The Epoch Times', 'Zionist Organization of America', 'The Catalyst', 'PRI (Public Radio International)', 'Scriberr Media - News', 'Andrew Sullivan', 'The Reporters Committee for Freedom of the Press', 'Splinter', 'Breitbart News', 'Walt Handelsman (cartoonist)', 'The Daily Wildcat', 'S.E. Cupp', 'New York Times - News', 'Time Magazine', 'CNS News', 'Rich Zeoli', 'Jonathan Chait', 'Michelle Goldberg', 'Arkansas Democrat-Gazette', 'Daily Northwestern', 'CivilPolitics.org', 'Fast Company', 'The Observer (New York)', 'Matt Towery', 'RealClearPolitics', 'New York Times (Online News)', 'Thomas Frey', 'John Fund', 'Politico', 'Gizmodo', 'Concord Monitor', 'Richard A. Lowry', 'The Verge', 'Quillette', 'Patriot Post', 'AARP', 'Peggy Noonan', 'The Nation', 'Rem Reider', 'BuzzFeed News', "Investor's Business Daily", 'NASA', 'Jezebel', 'Brown Girl Magazine', 'CBC News', 'Erraticus', 'NPR Editorial ', 'Red State', 'Joan Blades', 'The Western Journal', 'GLAAD', 'RAND Corporation', 'Human Rights Watch', 'NBC (Web News)', 'Tucson Weekly', 'Newsmax (News)', 'Jeff Jacoby', 'Whitehouse.gov', 'Brookings Institution', 'The New Yorker', 'International Institute for Strategic Studies', 'Billy Binion', 'Roll Call', 'Pacific Research Institute', 'Commentary Magazine', 'Jonathan Miller', 'Student Press Law Center', 'HotAir', 'Rev. Jesse Jackson Sr.', 'MediaPost', 'Mitch McConnell', 'Media Matters', 'Mitt Romney', 'Media Research Center', 'Black Enterprise', 'The Intercept', 'Bipartisan Policy Center', 'The Colorado Sun', 'Right Side News', 'Counter Currents', 'Business Insider', 'Associated Press', 'Yellow Scene Magazine', 'Ted Rall (cartoonist)', 'AZ Central', 'The American Mind', 'NewsBusters', 'FEE.org', 'Daily Mail', 'AllSides', 'NewsOne', 'Heavy.com', 'Star News', 'Howard Kurtz', 'Barack Obama', 'Michael Brendan Dougherty', 'CBN', 'Dana Milbank', 'Mother Jones', 'MIT News', 'Jeff Spross', 'Religion News Service', 'AP Fact Check', 'New York Times - Opinion', 'Elizabeth Warren', 'The Texan', 'CalMatters', 'Daily Cardinal', 'Reason Foundation', 'The Daily Caller', 'Newt Gingrich', 'The Root', 'Matt Welch', 'London School of Economics', 'The Heritage Foundation', 'Guest Writer - Center', 'Indy Online', 'San Gabriel Valley Tribune', 'The Jerusalem Post', 'MSNBC', 'Fox News (Online)', 'David Williamson', 'Wausau Daily Herald', 'Live Action News', 'Bloomberg', 'John Boehner', 'NMPolitics.net', 'John Gable, AllSides Founder', 'Frank Bruni', 'The Week - Opinion', 'Drudge Report', 'New York Magazine', 'Independent Institute', 'City Journal', 'The Libertarian Republic', 'Raw Story', 'CNN (Web News)', 'Committee to Protect Journalists', 'ZeroHedge', 'Paul Ryan', 'Kitsap Sun', 'The Dallas Morning News', 'Gallup', 'Michael Kinsley', 'WGN', 'Eli Parser', 'WFAE', 'Rutgers Today', 'DesMoines Register', 'Federation of American Scientists', 'The Hollywood Reporter', 'New York Post (Opinion)', 'Scriberr Media - Opinion/Editorial', 'Wired', 'Henry A. Brechter', 'CalWatchdog', 'Herald Democrat', 'The Commercial Appeal', 'Daily Chela', 'Current Affairs', 'Penn Live', 'Aaron Carroll', 'Allysia Finley (Wall Street Journal)', 'Mashable', 'Gail Collins', 'ESPN', 'Portland Press Herald', 'The American Spectator', 'Brent Bozell', 'Baltimore Sun', 'Ezra Klein', 'Vice', 'Quinnipiac University', 'TheBlaze.com', 'Star Tribune', 'CNBC', 'Ann Coulter', 'Quartz', 'Christopher Buskirk', 'Newsmax - Opinion', 'Orange County Register', 'The Seattle Times', 'Amy Klobuchar', 'The Dispatch', 'Center For American Progress', 'Fortune', 'Carnegie Endowment for International Peace', 'Wikipedia', 'PACE', 'FAIR', 'RollingStone.com', 'Foreign Policy', 'Jonathan Haidt', 'Esquire', 'Christian Science Monitor', 'The Texas Tribune', 'Joe Scarborough', 'Duke Chronicle', 'Center for Public Integrity', 'Above The Law', 'George Will', 'Bipartisan Report', 'Thomas Frank', 'MichelleMalkin.com', 'PJ Media', 'Smithsonian Magazine', 'Peacock Panache', 'Scientific American', 'NBCNews.com', 'Victor Hanson', 'Kellogg Insight', 'Tom Nichols', 'Oxford University Press', 'The Resurgent', 'Lawrence Journal World', 'Longmont Times-Call', 'Michael Goodwin', 'Block Club Chicago', 'Medical Daily', 'University of Wyoming', 'Misinformation Review', 'American Enterprise Institute', 'Daily Sabah', 'Ben Shapiro', 'Cato Institute (blog)', 'City Lab', 'Business Wire', 'Western Journal', 'Boston Herald', 'The Independent', 'Guest Writer - Left', 'Global Research', 'Julian Zelizer', 'National Journal', 'CU Independent', 'CNSNews.com', 'Mercatus Center', 'Louisville Courier-Journal', 'KUTV', 'Daily Beast', 'Arkadi Gerney', 'Andrew Napolitano', 'Newsmax', 'Scott Walker', 'ABC News', 'Pat Buchanan', 'PinkNews', 'The Hoya', 'LGBTQ Nation', 'Wall Street Journal - News', 'Bret Stephens', 'David Brooks', 'Neil J. Young', 'The Daily Wire', 'TechCrunch', 'Association for Psychological Science', 'New York Post (News)', 'The Boston Globe', 'RedCross.org', 'Charles Krauthammer', 'Ralph Benko', 'Rand Paul', 'ProPublica', 'NBC Today Show', 'Boston Review', 'Center - Major Media Sources', 'Guest Writer - Right', 'Detroit Free Press', 'Freakonomics', 'Washington Post', 'Des Moines Register', 'NBC News (Online)', 'Michael Barone', 'Jon Terbush', 'Chicago Sun-Times', 'American Greatness', 'National Review', 'Foreign Affairs', 'CBS SFBayArea', 'Yahoo! News', 'Los Angeles Times', 'National Interest', 'Conor Friedersdorf', 'Cato Institute', 'JSTOR Daily', 'Detroit News', 'CBS News', 'Marc A. Thiessen', 'Polish Times', 'The Globe and Mail', 'Metrocosm', 'San Francisco Chronicle', 'Cook Report', 'Bring Me The News', 'Damon Linker', 'SFGate', 'The Diplomat', 'Deseret News', 'Tucker Carlson', 'Mic', 'Chris Ruddy', 'The Guardian', 'Juan Williams', 'Fiscal Times', 'Latino Rebels', 'Washington Free Beacon', 'The Hill', 'Charles Blow', 'The Intelligencer', 'The Week - News', 'Bridgemi.com', 'The Justice', 'Mitú', 'Bucks County Courier Times', 'Carol Costello', 'Jesse Jackson', 'Vanity Fair', 'AL.com', 'Yahoo! The 360', 'Grist', 'Education Week', 'Psypost', 'Christianity Today', 'Deadline.com', 'MarketWatch', 'Townhall', 'Socialist Alternative', 'The Times of Israel', 'The Daily Texan', 'Democracy Now', 'Tallahassee Democrat', 'Jonah Goldberg', 'AZO Cleantech', 'The State Press', 'John Stossel', 'Tim Kaine', 'American Thinker', 'Brown University', 'Piers Morgan', 'OpenSecrets.org', 'BET', 'Washington Examiner', 'Socialist Project/The Bullet', 'Family Research Council', 'Noah Rothman', 'PBS NewsHour', 'San Jose Mercury News', 'Fox Online News', 'Atlanta Journal-Constitution', 'Fox News', 'FXStreet', 'Rasmussen Reports', 'Lifehacker', 'Michelle Malkin', 'Kathleen Parker', 'TED', 'The Telegraph - UK', 'The Post Millennial', 'University of Bath', 'Prager University', 'Nicholas Kristof', 'Teen Vogue', 'ACLU', 'The Economist', 'William Bennett', 'Conservative HQ', 'Fox News Opinion', 'International Business Times', 'Independent Journal Review', 'Inside Philanthropy', 'Voice of America', 'The American Conservative', 'Americans for Tax Reform', 'USA TODAY', 'Kalamazoo Gazette', 'KSL', 'TruthOut', 'Sally Pipes', 'CNN - Editorial', 'InfoWars', 'Marijuana Moment', 'Reuters', 'The Courier-Journal', 'The Thread', 'Ryan Cooper', 'Pew Research Center', 'Daily Camera', 'FiveThirtyEight', 'Scott Jennings', 'Houston Chronicle', 'Blue Virginia', 'Indiana Daily Student', 'Odyssey Online', 'Salon', 'Thomas Sowell', 'Washington Times', 'Jacobin', 'Las Vegas Sun', 'Bay Area Bandwith', 'Annafi Wahed', 'Commonwealth Journal', 'HuffPost', 'Jim Obergefell', 'Manhattan Institute', 'Webster Journal', 'Dick Morris', 'The Korea Herald', 'Rich Lowry', 'Chicago Tribune', 'New York Post', 'Defense One', 'Living Room Conversations', 'theunion.com', 'The Marshall Project', 'Fabius Maximus', 'The Atlantic', 'James Bovard', 'Al Jazeera', 'Hillary Clinton', 'Cosmopolitan', 'Slate', 'DAG Blog', 'USAPP', 'NPR Online News', 'Snopes', 'The Advocate', 'Fox News Latino', 'Boston Herald Editorial', 'Intellectual Conservative', 'New Republic', 'David Leonhardt', 'BBC News', 'WBFO', 'Military Times', 'Axios', 'ThinkProgress', 'East Bay Times', 'AlterNet', 'American Spectator', 'The Outline', 'Daily Kos']
    # for news in newses:
    #     print(news)
    
    dataset = CustomDatasetHyperpartisan("../../data/hyperpartisan/", "meta-llama/Llama-3.3-70B-Instruct")
    
# {
#     'topic': str,
#     'source': str,
#     'bias': int,
#     'url': str,
#     'title': str,
#     'date': str,
#     'authors': str,
#     'content': str,
#     'content_original': str,
#     'source_url': str,
#     'bias_text': str,
#     'ID': str
# }