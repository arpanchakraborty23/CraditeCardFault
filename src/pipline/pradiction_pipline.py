import os,sys
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from src.utils.utils import load_obj
import pdb

class PradictPipline:
    def __init__(self) -> None:
        logging.info('initialize object')

    def predict(self,features):
        try:
            # pdb.set_trace() 
            preprocessor_path=os.path.join("preprocess","preprocess.pkl")
            model_path=os.path.join("model/model.pkl")
            
            logging.info('object creating started')

            preprocessor=load_obj(preprocessor_path)
            model=load_obj(model_path)

            if preprocessor is not None:
                preprocessor.transform()
            else:
                print("Warning: your_object is None")


            scaled_fea=preprocessor.transform(features)
            pred=model.predict(scaled_fea)

            return pred

            logging.info('preadiction completed')

            return  pred


        except Exception as e:   
            logging.info('Error in Prediction') 
            logging.info(f"An error occurred: {str(e)}")    
            raise CustomException(sys,e) from e

class CustomData:
    def __init__(self, PAY_0:float,  PAY_2:float , PAY_3:float,  PAY_4:float,  PAY_5:float,  PAY_6:float,  
                       BILL_AMT1:float,  BILL_AMT2:float, BILL_AMT3:float,  BILL_AMT4:float,  BILL_AMT5:float,  BILL_AMT6:float, 
                       PAY_AMT1:float,  PAY_AMT2:float,  PAY_AMT3:float , PAY_AMT4:float , PAY_AMT5:float,  PAY_AMT6:float) -> None:
                       self.PAY_0=PAY_0,
                       self.PAY_2=PAY_2,
                       self.PAY_3 =PAY_3,
                       self.PAY_4= PAY_4 ,
                       self.PAY_5= PAY_5 ,
                       self.PAY_6 =PAY_6, 

                       self.BILL_AMT1= BILL_AMT1,  
                       self.BILL_ATM2=BILL_AMT2,
                       self.BILL_AMT3=BILL_AMT3,
                       self.BILL_AMT4 = BILL_AMT4,
                       self.BILL_AMT5= BILL_AMT5,
                       self.BILL_AMT6= BILL_AMT6,

                       self.PAY_AMT1=PAY_AMT1,
                       self.PAY_AMT2= PAY_AMT2,
                       self.PAY_AMT3= PAY_AMT3,
                       self.PAY_AMT4= PAY_AMT4,
                       self.PAY_AMT5= PAY_AMT5,
                       self.PAY_AMT6= PAY_AMT6

    def get_data_as_dataframe(self):
        try:
            logging.info('get_data as data frame')
            custom_data_dict={
                'PAY_0':[self.PAY_0],
                'PAY_2':[self.PAY_2],
                'PAY_3':[self.PAY_3], 
                'PAY_4':[self.PAY_4], 
                'PAY_5':[self.PAY_5], 
                'PAY_6':[self.PAY_6], 
                
                'BILL_AMT1':[self.BILL_AMT1],
                'BILL_AMT2':[self.BILL_ATM2],  
                'BILL_AMT3':[self.BILL_AMT3], 
                'BILL_AMT4':[self.BILL_AMT4],  
                'BILL_AMT5':[self.BILL_AMT5],  
                'BILL_AMT6':[self.BILL_AMT6], 

                'PAY_AMT1':[self.PAY_AMT1],  
                'PAY_AMT2':[self.PAY_AMT2],  
                'PAY_AMT3':[self.PAY_AMT3],  
                'PAY_AMT4':[self.PAY_AMT4],  
                'PAY_AMT5':[self.PAY_AMT5],  
                'PAY_AMT6':[self.PAY_AMT6]
            }          
            df=pd.DataFrame(custom_data_dict)
            logging.info('Read df in pradiction pipline completed')

            return df

        except Exception as e:
            logging.info(f'error occured in pradiction pipline df {str(e)}') 
            CustomException(sys,e)   
        

