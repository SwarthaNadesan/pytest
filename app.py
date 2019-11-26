import os
import io 
import base64


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



from flask import Flask,render_template,request
from werkzeug import secure_filename
from wordcloud import WordCloud


app = Flask(__name__)


# load the model from disk
clf = pickle.load(open('model.pkl', 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(filename)
        df = pd.read_csv(filename, encoding="latin-1")
        data= df['text']
        vect = cv.transform(data)        
        df.insert (2, "target", clf.predict(vect))
        os.remove(filename)
        #df.to_csv("com4.csv")
        swarm_plot =  sns.countplot(x='target',data=df)
        fig = swarm_plot.get_figure()     
        figfile = io.BytesIO()
        fig.savefig(figfile, format='png')     
        figfile.seek(0)    
        data_uri = base64.b64encode(figfile.read()).decode('ascii') 
        text = df.text[0]
        # Create and generate a word cloud image:
        filename = "cloud.png"
        wordcloud = WordCloud().generate(text)
        wordcloud.to_file(filename)


        with open(filename, "rb") as f:
             worldcloudimg=base64.b64encode(f.read()).decode('ascii') 
             os.remove(filename)   
    return render_template('index.html', plot_img =data_uri,wordcloud_img=worldcloudimg)


if __name__ == '__main__':
    app.run(debug=True)