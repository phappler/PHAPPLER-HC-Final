from flask import Flask, render_template, url_for, flash, redirect
from forms import SAForm
from sentimentanalysis import analyzesentiment
app = Flask(__name__)

app.config['SECRET_KEY'] = 'b73a5e5cbea0222d9ffb3d242fde285b'

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def hello():
   form = SAForm()
   if form.validate_on_submit():
      print("hello")
      return redirect(url_for("results", file=form.file.data, split=form.resplit.data))
   return render_template('home.html', form=form)
   
@app.route("/about")
def about():
   return render_template('about.html', title='About')

@app.route("/results/<file>/<split>")
def results(file, split):
   res = analyzesentiment(file, split)
   pos = res["possent"]
   neg = res["negsent"]
   sec = res["sections"]
   positive = []
   for k in pos:
      positive.append((k, sec[k[1]]))
   negative = []
   for k in neg:
      negative.append((k, sec[k[1]]))
   return render_template('results.html', title='Results', positive=positive, negative=negative)
 
if __name__ == '__main__':
   app.run(debug=True)