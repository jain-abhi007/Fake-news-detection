import csv
from flask import render_template,request,redirect
from app import app
from app.forms import SubmitForm
@app.route('/')

@app.route('/index',methods=['GET','POST'])
def index():
    
    form=SubmitForm()
    if request.method == "POST":
        abc=request.form['query']
        print(abc)
        csv_file=open('data.csv','w')
        writer=csv.writer(csv_file)
        writer.writerow(abc)
        redirect('/')
    return render_template('index.html',form=form)
