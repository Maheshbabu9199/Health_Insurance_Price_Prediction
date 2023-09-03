from flask import Flask, render_template, request
import pandas as pd

from src.pipelines.Predict_Pipeline import CustomData, PredictPipeline

application = Flask(__name__)



@application.route("/")
def index():
    return "<h2>Index Loaded</h2>"



@application.route("/homepage", methods=["GET", "POST"])
def homepage():
    if request.method == "GET":
        return render_template("homepage.html")
    else:
        age = request.form.get('age')
        gender = request.form.get('gender')
        region = request.form.get('region')
        smoker = request.form.get('smoker')
        bmi = request.form.get('bmi')
        children = request.form.get('children')

        custom_data_obj = CustomData(age, gender, bmi, region, children, smoker)
        process_data = custom_data_obj.get_dataframe()
        prediction = PredictPipeline(process_data)
      

        return render_template("homepage.html", expense = prediction.expense[0])
        #return "<h2 style='text-align:center'> {}</h2>".format(prediction.expense)



if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)