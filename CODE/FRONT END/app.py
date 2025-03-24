
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from flask import Flask, render_template, request, session
import pymysql
import pandas as pd
from prophet import Prophet
from datetime import datetime
from flask import redirect, url_for


app = Flask(__name__)
app.secret_key = "crypto"
data_loaded = False  
df = None  

# Database connection
db = pymysql.connect(
    host='localhost',
    user='root',
    password='root',
    port=3306,
    database='crypto'
)
cur = db.cursor()

@app.route('/')
def index():
    return render_template("Index.html")

@app.route('/Login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        useremail = request.form['useremail']
        session['useremail'] = useremail
        userpassword = request.form['userpassword']
        sql = "SELECT COUNT(*) FROM user WHERE Email=%s AND Password=%s"
        cur.execute(sql, (useremail, userpassword))
        count = cur.fetchone()[0]
        
        if count == 0:
            return render_template("Login.html", name="User credentials are not valid")
        
        sql = "SELECT * FROM user WHERE Email=%s AND Password=%s"
        cur.execute(sql, (useremail, userpassword))
        user_data = cur.fetchone()
        
        session['email'] = useremail
        session['pno'] = str(user_data[4])
        session['name'] = str(user_data[1])
        return render_template("Userhome.html", myname=session['name'])
    
    return render_template('Login.html')

@app.route('/new')
def new():
    return render_template("Userhome.html", myname=session['name'])
    
@app.route('/Logout')
def logout():
    global df, data_loaded, selected_model, selected_model_name, model_accuracy, x_train, y_train, feature_names
    
    # Clear session data
    session.clear()
    
    # Reset dataset and model-related variables
    df = None
    data_loaded = False
    selected_model = None
    selected_model_name = ""
    model_accuracy = 0
    x_train, y_train = None, None
    feature_names = []

    return redirect(url_for('index'))  # Redirect to the homepage


@app.route('/Load data', methods=['POST', 'GET'])
def load_data():
    global df, data_loaded
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            data_loaded = True
            return render_template('Load data.html', msg="Data successfully loaded.", data_loaded=data_loaded)
    
    return render_template('Load data.html', msg="Please upload a dataset.", data_loaded=data_loaded)

@app.route('/View data')
def view_data():
    global df,data_loaded  

    if df is None or df.empty:
        msg = "No data loaded. Please load data first."
        return render_template('View data.html', msg=msg, col_name=[], row_val=[])

    return render_template('View data.html', col_name=df.columns, row_val=df.values.tolist(), msg=None,data_loaded=data_loaded)

selected_model = None  
selected_model_name = ""  
model_accuracy = 0  
x_train, y_train = None, None  
feature_names = []  

@app.route('/Model', methods=['GET', 'POST'])
def model():
    global df, data_loaded, selected_model, selected_model_name, model_accuracy, x_train, y_train, feature_names

    if not data_loaded:
        return render_template('Model.html', msg="Please load data first.", data_loaded=data_loaded)

    if request.method == "POST":
        model_choice = int(request.form['selected'])
        
        X = df.drop(columns=['Pred'], axis=1)
        y = df['Pred']
        feature_names = X.columns.tolist()  
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        classifiers = {
            1: (LinearDiscriminantAnalysis(), "Linear Discriminant Analysis"),
            2: (MLPClassifier(max_iter=200), "MLP Classifier"),
            3: (ExtraTreesClassifier(n_jobs=-1), "Extra Trees Classifier"),
            4: (VotingClassifier(estimators=[('mlp', MLPClassifier(max_iter=100)), 
                                             ('etc', ExtraTreesClassifier(n_jobs=-1))], voting='hard'), "Hybrid Model"),
        }

        if model_choice in classifiers:
            clf, name = classifiers[model_choice]
            clf.fit(x_train, y_train)
            selected_model = clf  
            selected_model_name = name
            y_pred = clf.predict(x_test)
            model_accuracy = accuracy_score(y_test, y_pred) * 100

            return render_template('Model.html', 
                                   msg=f'The accuracy obtained by {selected_model_name} is {model_accuracy:.2f}%', 
                                   data_loaded=data_loaded)

    return render_template('Model.html', data_loaded=data_loaded)

@app.route('/Prediction', methods=["POST", "GET"])
def prediction():
    global selected_model, selected_model_name, model_accuracy, feature_names, data_loaded

    if selected_model is None:
        return render_template("Prediction.html", msg="Please select and train a model first.", data_loaded=data_loaded)

    if request.method == "POST":
        try:
            features = [float(request.form.get(key, 0)) for key in feature_names]

            if len(features) != len(feature_names):
                return render_template("Prediction.html", 
                                       msg=f"Error: Expected {len(feature_names)} features, but got {len(features)}.", 
                                       data_loaded=data_loaded)

            features = np.array(features).reshape(1, -1)
            prediction = selected_model.predict(features)[0]
            result = "Risk Found" if prediction == 1 else "No Risk Found"

            return render_template('Prediction.html', 
                                   msg=f"Financial Risk-Type: {result}", 
                                   model_name=selected_model_name, 
                                   accuracy=f"Model Accuracy: {model_accuracy:.2f}%", 
                                   data_loaded=data_loaded)
        except Exception as e:
            return render_template("Prediction.html", msg=f"Error in prediction: {str(e)}", data_loaded=data_loaded)

    return render_template("Prediction.html", data_loaded=data_loaded)

@app.route('/Bitcoin')
def Bitcoin():
    return render_template('Bitcoin.html')

file_path = "BTC-USD.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

model = Prophet()
model.fit(df)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/viewdata')
def viewdata():
    df = pd.read_csv('BTC-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('viewdata.html', table=table_html)

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/predictionnew', methods=['GET', 'POST'])
def predictionnew():
    forecast = pd.DataFrame()  
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        days_diff = (end_date - start_date).days
        if days_diff <= 0:
            return render_template('predictionnew.html', forecast=[])

        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        future = pd.DataFrame(future_dates, columns=['ds'])

        forecast = model.predict(future)

    return render_template('predictionnew.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])

if __name__ == '__main__':
    app.run(debug=True)

