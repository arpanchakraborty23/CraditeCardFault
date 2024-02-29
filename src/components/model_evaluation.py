


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    obj=DataTransormation()
    train_arr,test_arr,_=obj.initiate_transformaation(train_data,test_data)

    obj=ModelTrain()
    print(obj.initate_model_train(train_arr,test_arr))