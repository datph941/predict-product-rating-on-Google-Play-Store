from unicodedata import category
from flask import Flask, escape, request, render_template
import  pickle
from xgboost import XGBRegressor
import numpy as np
# import oss

app = Flask(__name__)

# port
# port = int(os.environ.get("PORT", 5000))

# app.run(host='0.0.0.0', port=port, debug=True)

model = pickle.load(open('rating_pred_tree', 'rb'))
model1 = pickle.load(open('rating_pred_xgb', 'rb'))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        category = float(request.form.get("category"))
        rating_count = float( request.form.get("rating_count"))
        installs =  float(request.form.get("installs"))
        minimum_installs =  float(request.form.get("minimum_installs"))
        maximum_installs =  float(request.form.get("maximum_installs"))
        free =  float(request.form.get("free"))
        price =  float(request.form.get("price"))
        size =  float(request.form.get("size"))
        minimum_android =  float(request.form.get("minimum_android"))
        content_rating =  float(request.form.get("content_rating"))
        ad_supported =  float(request.form.get("ad_supported"))
        in_app_purchases =  float(request.form.get("in_app_purchases"))
        editors_choice =  float(request.form.get("editors_choice"))
        data = [category, rating_count, installs, minimum_installs, maximum_installs, free, price, size, minimum_android, content_rating, ad_supported, in_app_purchases, editors_choice]
        data = np.array(data).reshape(1, -1)
        dataInput = [category, rating_count, installs, minimum_installs, maximum_installs, free, price, size, minimum_android, content_rating, ad_supported, in_app_purchases, editors_choice]
        prediction = model.predict(data)
        prediction1 = model1.predict(data)
        output = float(prediction[0])
        output1 = float(prediction1[0])
        return render_template("result.html", prediction_text=output, prediction_text1=output1, dataInput=dataInput)
        # return render_template("index.html", prediction_text1="The app is {}".format(output))
    else:
        return render_template('index.html')

@app.route("/predict")
def predict():
    return render_template('result.html')

if __name__ == "__main__":
    app.run()