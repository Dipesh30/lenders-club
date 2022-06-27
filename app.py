from sklearn.preprocessing import StandardScaler,QuantileTransformer
import joblib
from utils import read_config,load_bins
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger


def std_num_cols(df,num_cols,scalar):
    df[num_cols] = scalar.transform(df[num_cols])
    return df

def one_hot_cat_cols(df,cat_cols,cat_dict):
    results = []
    for col_name in cat_cols:
        known_cats = cat_dict[col_name]
        df_cat = pd.Categorical(df[col_name].values, categories = known_cats)
        df_cat = pd.get_dummies(df_cat,prefix=col_name)
        results.append(df_cat)
    df_cat_ = pd.concat(results,axis=1).reset_index(drop=True)
    return df_cat_

def preprocess_data(df,num_cols,cat_cols,scalar,cat_dict):
    #df = feat_eng(df)
    df = df.drop(columns=['address'])
    df = std_num_cols(df,num_cols,scalar)
    df_cat = one_hot_cat_cols(df,cat_cols,cat_dict)
    dff = pd.concat([df[num_cols].reset_index(drop=True),
               df_cat.reset_index(drop=True)],axis=1)

    return dff

def predict(df,model,scalar_dict,cat_dict):
    
    num_cols =['loan_amnt','int_rate','installment','annual_inc','dti','open_acc','pub_rec',
           'revol_bal','revol_util','total_acc','mort_acc','pub_rec_bankruptcies']
    
    cat_cols =  ['term','grade','sub_grade','emp_length','home_ownership','verification_status','application_type']
    
    label_dict = {1:"Fully Paid",0:"Charged Off"}
    x = preprocess_data(df,num_cols,cat_cols,scalar_dict,cat_dict)
    
    out = model.predict(x)[0]
    pred_label = label_dict[out]
    pred_prob = model.predict_proba(x)[0]
    return out,pred_label,pred_prob

In = []

fastapp =  FastAPI()
loaded_rf = joblib.load("./random_forest.joblib")
scalar = joblib.load('./nul_col.bin')
cat_dict = joblib.load('./cat_col.bin')

class predictRequest(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    mort_acc: float
    pub_rec_bankruptcies: float
    term: str
    grade: str
    sub_grade: str
    emp_title: str
    emp_length: str
    home_ownership: str
    verification_status: str
    issue_d: str
    purpose: str
    title: str
    earliest_cr_line: str
    initial_list_status: str
    application_type: str
    address: str
        
@fastapp.post("/Input")
def Input(request: predictRequest):
    logger.debug(request)
    In.append(request.dict())
    
    predict_date = pd.DataFrame(test_data,index=[0])
    print(predict_date)
    
    # = preprocess_data(predict_date)
    
    out,pred_label,pred_prob = predict(predict_date,loaded_rf,scalar,cat_dict)    
    prob = round(pred_prob[1],2)
        
    return {"predicted_class":str(out),
            "predicted_label":str(pred_label),
            "predicted_probability":str(prob)}

web: uvicorn app:fastapp --host=0.0.0.0 --port=${PORT:-5000}
