from flask import Flask,render_template,request,send_file,jsonify
from src.exception import CustomException
from src.logger import logging
from src.pipline.bulk_pradiction import PredictionPipeline

import sys

app= Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        logging.info('prediction started')
        if request.method=='POST':
            prediction_pipeline = PredictionPipeline()


            prediction_file_detail= prediction_pipeline.run_pipline()

            logging.info('Prediction Completed Download file')

            return send_file(prediction_file_detail.prediction_file_path,download_name=prediction_file_detail.prediction_file_name,as_attachment=True)



        else:
            return render_template('upload_file.html')

    except Exception as e:
        logging.info(f'error in bulk app prediction {str(e)}')
        raise CustomException(sys,e) from e 

if __name__=="__main__":
    app.run(debug=True,port=5000)
