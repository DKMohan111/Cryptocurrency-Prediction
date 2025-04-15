

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
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, jsonify


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
@app.route('/user')
def user():
    global df,data_loaded 
    return render_template('userhome.html', data_loaded=data_loaded,myname=session['name'])

    

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



@app.route('/Registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        contact = request.form['contact']
        
        if userpassword == conpassword:
            print(9999999999999999)
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            print(5555, data)
            
            if data:
                print(7777777777777777)
                msg="Credentials already exist!"
                return render_template("Registration.html",msg=msg)
            
            else:
                print(88888888888888)
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully"
                return render_template("Login.html",msg=msg)
                
        else:
            print(666666666666666)
            msg="Password doesn't match"
            return render_template("Registration.html",msg=msg)
    return render_template('Registration.html')
    
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
    global df, selected_model, selected_model_name, model_accuracy, feature_names, data_loaded

    if not data_loaded:
        return render_template('Model.html', msg="Please load data first.", data_loaded=data_loaded)

    target_column = 'Pred' if 'Pred' in df.columns else None

    if request.method == "POST":
        model_choice = int(request.form['selected'])
        selected_target = request.form.get('target_column', target_column)

        if not selected_target or selected_target not in df.columns:
            return render_template('Model.html', msg="Please select a valid target column.", data_loaded=data_loaded)

        X = df.select_dtypes(include=[np.number]).drop(columns=[selected_target], errors='ignore')
        y = df[selected_target]
        feature_names = X.columns.tolist()

        if X.empty:
            return render_template('Model.html', msg="No valid numerical features found in dataset.", data_loaded=data_loaded)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        models = {
            1: (LinearDiscriminantAnalysis(), "Linear Discriminant Analysis"),
            2: (MLPClassifier(max_iter=200), "MLP Classifier"),
            3: (ExtraTreesClassifier(n_jobs=-1), "Extra Trees Classifier"),
            4: (VotingClassifier(estimators=[('mlp', MLPClassifier(max_iter=100)), 
                                             ('etc', ExtraTreesClassifier(n_jobs=-1))], voting='hard'), "Hybrid Model"),
        }

        if model_choice not in models:
            return render_template('Model.html', msg="Invalid model selection.", data_loaded=data_loaded)

        clf, name = models[model_choice]
        clf.fit(x_train, y_train)
        selected_model = clf
        selected_model_name = name
        y_pred = clf.predict(x_test)
        model_accuracy = accuracy_score(y_test, y_pred) * 100

        return render_template('Model.html', 
                               msg=f'{selected_model_name} Accuracy: {model_accuracy:.2f}%', 
                               data_loaded=data_loaded)

    # Provide a dropdown for target column selection if 'Pred' is missing
    target_options = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return render_template('Model.html', data_loaded=data_loaded, target_options=target_options)


@app.route('/Advanced', methods=['GET', 'POST'])
def advanced():
    global df, selected_model, selected_model_name, model_accuracy, data_loaded

    # 🛠 Ensure dataset is loaded
    if df is None or "name" not in df.columns:
        return render_template("Advanced.html", 
                               msg="No dataset loaded. Please load data first.", 
                               coins=[], 
                               data_loaded=data_loaded)

    # 🛠 Ensure model is selected before proceeding
    if selected_model is None:
        return render_template("Advanced.html", 
                               msg="No model selected. Please select a model first.", 
                               coins=[], 
                               data_loaded=data_loaded)

    # 🛠 Get list of available cryptocurrencies
    coins = df["name"].unique().tolist()

    if request.method == "POST":
        selected_crypto = request.form.get("crypto_name")

        if not selected_crypto:
            return jsonify({"msg": "Please select a cryptocurrency.", "error": True})

        crypto_data = df[df["name"] == selected_crypto].drop(columns=["name", "Pred"], errors="ignore")

        if crypto_data.empty:
            return jsonify({"msg": f"No data found for {selected_crypto}.", "error": True})

        features = crypto_data.iloc[0].values.reshape(1, -1)

        try:
            prediction = selected_model.predict(features)[0]
            result = "Risk Found" if prediction == 1 else "No Risk Found"
            return jsonify({
                "msg": f"Financial Risk-Type: {result} for {selected_crypto}",
                "model_name": selected_model_name,
                "accuracy": f"Model Accuracy: {model_accuracy:.2f}%"
            })
        except Exception as e:
            return jsonify({"msg": f"Error in prediction: {str(e)}", "error": True})

    # 🛠 Render the page for GET request
    return render_template("Advanced.html", coins=coins, data_loaded=data_loaded)

@app.route('/get_crypto_data', methods=['POST'])
def get_crypto_data():
    global df
    
    try:
        data = request.get_json()  # Read JSON data from AJAX request
        selected_crypto = data.get("crypto_name")

        if not selected_crypto:
            return jsonify({"error": True, "msg": "No cryptocurrency selected"})

        if df is None or "name" not in df.columns:
            return jsonify({"error": True, "msg": "No dataset loaded. Please load data first."})

        # Filter the selected cryptocurrency data
        crypto_data = df[df["name"] == selected_crypto]

        if crypto_data.empty:
            return jsonify({"error": True, "msg": f"No data found for {selected_crypto}."})

        # Convert first row to JSON
        crypto_data = crypto_data.iloc[0].to_dict()

        return jsonify(crypto_data)

    except Exception as e:
        return jsonify({"error": True, "msg": f"Error fetching data: {str(e)}"})

@app.route('/Bitcoin')
def Bitcoin():
    return render_template('Bitcoin.html')

@app.route('/Ethereum')
def Ethereum():
    return render_template('Ethereum.html')

@app.route('/Solana')
def Solana():
    return render_template('Solana.html')

@app.route('/Binance')
def Binance():
    return render_template('Binance.html')

@app.route('/Ripple')
def Ripple():
    return render_template('Ripple.html')

@app.route('/Cardano')
def Cardano():
    return render_template('Cardano.html')

@app.route('/Avalanche')
def Avalanche():
    return render_template('Avalanche.html')

@app.route('/Dogecoin')
def Dogecoin():
    return render_template('Dogecoin.html')

@app.route('/Toncoin')
def Toncoin():
    return render_template('Toncoin.html')

@app.route('/Polkadot')
def Polkadot():
    return render_template('Polkadot.html')
@app.route('/Select')
def select_crypto():
    return render_template('Selector.html')

btc_df = pd.read_csv("BTC-USD.csv")
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df = btc_df[['Date', 'Close']]
btc_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
btc_model = Prophet()
btc_model.fit(btc_df)

eth_df = pd.read_csv("ETH-USD.csv")
eth_df['Date'] = pd.to_datetime(eth_df['Date'])
eth_df = eth_df[['Date', 'Close']]
eth_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
eth_model = Prophet()
eth_model.fit(eth_df)

sol_df = pd.read_csv("SOL-USD.csv")
sol_df['Date'] = pd.to_datetime(sol_df['Date'])
sol_df = sol_df[['Date', 'Close']]
sol_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
sol_model = Prophet()
sol_model.fit(sol_df)

BNB_df = pd.read_csv("BNB-USD.csv")
BNB_df['Date'] = pd.to_datetime(BNB_df['Date'])
BNB_df = BNB_df[['Date', 'Close']]
BNB_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
BNB_model = Prophet()
BNB_model.fit(BNB_df)

XRP_df = pd.read_csv("XRP-USD.csv")
XRP_df['Date'] = pd.to_datetime(XRP_df['Date'])
XRP_df = XRP_df[['Date', 'Close']]
XRP_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
XRP_model = Prophet()
XRP_model.fit(XRP_df)

ADA_df = pd.read_csv("ADA-USD.csv")
ADA_df['Date'] = pd.to_datetime(ADA_df['Date'])
ADA_df = ADA_df[['Date', 'Close']]
ADA_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
ADA_model = Prophet()
ADA_model.fit(ADA_df)

AVAX_df = pd.read_csv("AVAX-USD.csv")
AVAX_df['Date'] = pd.to_datetime(AVAX_df['Date'])
AVAX_df = AVAX_df[['Date', 'Close']]
AVAX_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
AVAX_model = Prophet()
AVAX_model.fit(AVAX_df)

DOGE_df = pd.read_csv("DOGE-USD.csv")
DOGE_df['Date'] = pd.to_datetime(DOGE_df['Date'])
DOGE_df = DOGE_df[['Date', 'Close']]
DOGE_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
DOGE_model = Prophet()
DOGE_model.fit(DOGE_df)

TON11419_df = pd.read_csv("TON11419-USD.csv")
TON11419_df['Date'] = pd.to_datetime(TON11419_df['Date'])
TON11419_df = TON11419_df[['Date', 'Close']]
TON11419_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
TON11419_model = Prophet()
TON11419_model.fit(TON11419_df)

DOT_df = pd.read_csv("DOT-USD.csv")
DOT_df['Date'] = pd.to_datetime(DOT_df['Date'])
DOT_df = DOT_df[['Date', 'Close']]
DOT_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
DOT_model = Prophet()
DOT_model.fit(DOT_df)





@app.route('/Bitcoindata')
def Bitcoindata():
    df = pd.read_csv('BTC-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Bitcoindata.html', table=table_html)

@app.route('/Ethereumdata')
def Ethereumdata():
    df = pd.read_csv('ETH-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Ethereumdata.html', table=table_html)

@app.route('/Solanadata')
def Solanadata():
    df = pd.read_csv('SOL-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Solanadata.html', table=table_html)

@app.route('/Binancedata')
def Binancedata():
    df = pd.read_csv('BNB-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Binancedata.html', table=table_html)

@app.route('/Rippledata')
def Rippledata():
    df = pd.read_csv('XRP-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Rippledata.html', table=table_html)

@app.route('/Cardanodata')
def Cardanodata():
    df = pd.read_csv('ADA-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Cardanodata.html', table=table_html)

@app.route('/Avalanchedata')
def Avalanchedata():
    df = pd.read_csv('AVAX-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Avalanchedata.html', table=table_html)

@app.route('/Dogecoindata')
def Dogecoindata():
    df = pd.read_csv('DOGE-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Dogecoindata.html', table=table_html)

@app.route('/Toncoindata')
def Toncoindata():
    df = pd.read_csv('TON11419-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Toncoindata.html', table=table_html)

@app.route('/Polkadotdata')
def Polkadotdata():
    df = pd.read_csv('DOT-USD.csv')
    table_html = df.tail(1000).to_html(classes='table table-striped table-hover', index=False)
    return render_template('Polkadotdata.html', table=table_html)


@app.route('/Bitcoinpred', methods=['GET', 'POST'])
def Bitcoinpred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Bitcoinpred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = btc_model.predict(future)

    return render_template('Bitcoinpred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Ethereumpred', methods=['GET', 'POST'])
def Ethereumpred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Ethereumpred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = eth_model.predict(future)

    return render_template('Ethereumpred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Solanapred', methods=['GET', 'POST'])
def Solanapred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Solanapred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = sol_model.predict(future)

    return render_template('Solanapred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])

@app.route('/Binancepred', methods=['GET', 'POST'])
def Binancepred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Binancepred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = BNB_model.predict(future)

    return render_template('Binancepred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Ripplepred', methods=['GET', 'POST'])
def Ripplepred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Ripplepred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = XRP_model.predict(future)

    return render_template('Ripplepred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Cardanopred', methods=['GET', 'POST'])
def Cardanopred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Cardanopred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = ADA_model.predict(future)

    return render_template('Cardanopred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Avalanchepred', methods=['GET', 'POST'])
def Avalanchepred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Avalanchepred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = AVAX_model.predict(future)

    return render_template('Avalanchepred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Dogecoinpred', methods=['GET', 'POST'])
def Dogecoinpred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Dogecoinpred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = DOGE_model.predict(future)

    return render_template('Dogecoinpred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Toncoinpred', methods=['GET', 'POST'])
def Toncoinpred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Toncoinpred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = TON11419_model.predict(future)

    return render_template('Toncoinpred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


@app.route('/Polkadotpred', methods=['GET', 'POST'])
def Polkadotpred():
    forecast = pd.DataFrame()
    if request.method == "POST":
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        if (end_date - start_date).days <= 0:
            return render_template('Polkadotpred.html', forecast=[])

        future = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'), columns=['ds'])
        forecast = DOT_model.predict(future)

    return render_template('Polkadotpred.html', forecast=forecast.to_dict(orient='records') if not forecast.empty else [])


if __name__ == '__main__':
    app.run(debug=True)

