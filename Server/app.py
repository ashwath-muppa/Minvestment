from flask import Flask, render_template, request, redirect, url_for
import torch
import requests
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from GoogleNews import GoogleNews
from bs4 import BeautifulSoup
import google.generativeai as genai
import requests
from polygon import RESTClient
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt

device = "cpu"
app = Flask(__name__)

def pred(input):
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load('./model_better', map_location=torch.device('cpu')))
    model.eval()
    text = input
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits)
    return predicted_class

def sent(x):
    x = x.split(".")
    y = []
    for i in range(len(x)):
        if len(x[i])<2000:
            y.append(x[i])
    k = 0
    for i in y:
        c = pred(i)
        k+=(c-1)
    return k

def func(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    article = soup.get_text(separator=' ', strip=True)
    if len(article)>=4001:
        article = article[:4000]
    z = sent(article)
    return z

def score(input):
    den = len(input)
    num = 0
    for i in input:
        num+=i
    return num/den

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_input = request.form['text_input']
        googlenews = GoogleNews()
        googlenews.enableException(True)
        googlenews = GoogleNews(lang='en')
        search = text_input
        search+=' company'
        googlenews.get_news(search)
        a = googlenews.get_links()
        ls = []
        for i in a[:5]:
            ls.append(func("https://"+i))
        news = score(ls)
        think = ""
        key = 'AIzaSyBneXq_g6QS1D6JnfczPkX9q3gKbvXYNSs'
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-1.0-pro-latest')
        if text_input.lower() == 'raytheon':
            tick = 'RTX'
        if text_input.lower() == 'netflix':
            tick = 'NFLX'
        else:
            response = model.generate_content(f'give me the stock ticker for {text_input}, and only give me the ticker, no other words')
            tick = response.text
        api_key = 'xaPm_yZ0eGk6oQGqfz5J1sIZwsAEdVbp'
        client = RESTClient(api_key)
        today = datetime.today()
        today_s = str(datetime.today())[:10]
        wka = datetime.today()-timedelta(days=5)
        week_ago_s = str(datetime.today()-timedelta(days=5))[:10]
        aggs = []
        for a in client.list_aggs(
            tick,
            1,
            "day",
            week_ago_s,
            today,
            limit=50000,
        ):
            aggs.append(a)
        dates = []
        opens = []
        closes = []
        highs = []
        lows = []
        for i in aggs:
            dates.append(str(wka)[:10])
            opens.append(i.open)
            closes.append(i.close)
            highs.append(i.high)
            lows.append(i.low)
            wka = wka + timedelta(days=1)
        plt.switch_backend('Agg')
        plt.plot(dates, opens)
        plt.plot(dates, closes)
        plt.plot(dates, highs)
        plt.plot(dates, lows)
        plt.title('High/Low/Open/Close Values')
        plt.legend(['Open', 'Close', 'High', 'Low'])
        plt.savefig('static/plot.png')
        new_prompt = f"for these stock statistics make a paragraph to analyze the trend. Use dollars as units, do not yap and be concise (limit to 200 words), make sure to include sharp turns and drops, don't use analogies: dates: {dates} \n open price: {opens} \n close price: {closes} \n high price: {highs} \n low price: {lows}"
        response2 = model.generate_content(new_prompt)
        x = round(news.tolist(),3)
        if x<=1 and x>=-1:
            sign = 'negative' if x<0 else 'positive'
            return redirect(url_for('resultneutral', news=str(x), think=response2.text, company=tick, sign = sign))
        elif x>1:
            sign='positive'
            return redirect(url_for('resultpositive', news=str(x), think=response2.text, company=tick))
        else:
            sign='negative'
            return redirect(url_for('resultnegative', news=str(x), think=response2.text, company=tick))
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/resultneutral')
def resultneutral():
    news = request.args.get('news')
    think = request.args.get('think')
    company = request.args.get('company')
    sign = request.args.get('sign')
    return render_template('resultneutral.html', news=news, think=think, company=company, sign=sign)


@app.route('/resultpositive')
def resultpositive():
    news = request.args.get('news')
    think = request.args.get('think')
    company = request.args.get('company')
    return render_template('resultpositive.html', news=news, think=think, company=company)

@app.route('/resultnegative')
def resultnegative():
    news = request.args.get('news')
    think = request.args.get('think')
    company = request.args.get('company')
    return render_template('resultnegative.html', news=news, think=think, company=company)

if __name__ == '__main__':
    app.run(debug=True)
