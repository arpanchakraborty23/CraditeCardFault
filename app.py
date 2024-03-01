from flask import Flask,request,render_template,jsonify
import sys
import pandas as pd

from src.pipline.pradiction_pipline import PradictPipline,CustomData
from src.logger import logging
from src.exception import CustomException

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    try:
        if request.method=="GET":
            return render_template('form.html')
        
        else:
            data=CustomData(
                PAY_0=request.form.get('PAY_0'),
                PAY_2=request.form.get('PAY_2'),  
                PAY_3=request.form.get('PAY_3'),  
                PAY_4=request.form.get('PAY_4'),  
                PAY_5=request.form.get('PAY_5'), 
                PAY_6=request.form.get('PAY_6'),  
                BILL_AMT1=request.form.get(' BILL_AMT1'), 
                BILL_AMT2=request.form.get(' BILL_AMT2'),   
                BILL_AMT3=request.form.get(' BILL_AMT3'),   
                BILL_AMT4=request.form.get(' BILL_AMT4'),   
                BILL_AMT5=request.form.get(' BILL_AMT5'),   
                BILL_AMT6=request.form.get(' BILL_AMT6'),   
                PAY_AMT1=request.form.get('PAY_AMT1'),   
                PAY_AMT2=request.form.get('PAY_AMT2'),
                PAY_AMT3=request.form.get('PAY_AMT3'),
                PAY_AMT4=request.form.get('PAY_AMT4'),
                PAY_AMT5=request.form.get('PAY_AMT5'),
                PAY_AMT6=request.form.get('PAY_AMT6')
            )
            
            final_data=data.get_data_as_dataframe()
            pred=prediction_pipline=PradictPipline()

            prediction_pipline.predict(final_data)

            result=round(pred[0],2)

            return render_template('result.html',final_result=result)
        
    except Exception as e:
        logging.info('error in pradict app')
        raise CustomException(e,sys)


if __name__=="__main__":
    app.run(port=5000)