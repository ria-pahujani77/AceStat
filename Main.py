import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def initial():
    index_col_names=['unit_id','time_cycle']
    operat_set_col_names=['oper_set{}'.format(i) for i in range(1,4)]
    sensor_measure_col_names=['sm_{}'.format(i) for i in range(1,22)]
    all_col=index_col_names+operat_set_col_names+sensor_measure_col_names
    initial1(all_col)

def initial1(all_col):
    org_df=pd.read_csv("C:\\Users\\asus\\Desktop\\AceStat\\train_FD001.txt" ,delim_whitespace=True ,names=all_col)
    org_df
    initial2(org_df, all_col)

def initial2(org_df, all_col):
    acc_rul=pd.read_csv("C:\\Users\\asus\\Desktop\\AceStat\\RUL_FD001.txt", delim_whitespace=True, names=["RUL"])
    acc_rul['unit_id']=acc_rul.index+1
    acc_rul.head()
    initial3(org_df, all_col)

def initial3(org_df, all_col):
    max_time_cycle=org_df.groupby('unit_id')['time_cycle'].max()
    rul = pd.DataFrame(max_time_cycle).reset_index()
    rul.columns = ['unit_id', 'max']
    rul.head()
    org_df=org_df.merge(rul, on=['unit_id'], how='left')
    initial4(org_df, all_col)

def initial4(org_df, all_col):
    org_df['RUL']= org_df['max']-org_df['time_cycle']
    org_df.drop('max', axis=1, inplace=True)
    initial5(org_df, all_col)

def initial5(org_df, all_col):
    useless_col=['oper_set3','sm_1','sm_5','sm_6','sm_10','sm_14','sm_16','sm_18','sm_19']
    train_df=org_df.drop(useless_col, axis=1)
    initial6(train_df)
    
def initial6(train_df):
    from sklearn.preprocessing import MinMaxScaler
    features=list(train_df.columns[1:-1])
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))

    train_df[features] = min_max_scaler.fit_transform(train_df[features])
    X = train_df.drop(['unit_id','RUL'],axis=1).values
    y = train_df['RUL'].values
    initial7(X, y)

def initial7(X, y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    X_train,x_val, y_train, y_val=train_test_split(X,y,test_size=0.2,random_state=42)
    reg = GradientBoostingRegressor(max_depth=5, n_estimators=500,random_state=42)
    reg.fit(X_train,y_train)
    initial8(reg, x_val, LinearRegression, r2_score, y_val, X_train, y_train)

def initial8(reg, x_val, LinearRegression, r2_score, y_val, X_train, y_train):
    y_pred = reg.predict(x_val)
    initial9(reg, x_val, LinearRegression, r2_score, y_val, y_pred, X_train, y_train)

def initial9(reg, x_val, LinearRegression, r2_score, y_val, y_pred, X_train, y_train):
    print(r2_score(y_val,y_pred))
    reg2=LinearRegression()
    reg2.fit(X_train,y_train)
    y_pred2=reg2.predict(x_val)
    print(r2_score(y_val,y_pred2))
    import pickle
    with open('model.pkl', 'wb') as files:
        pickle.dump(reg, files)
    with open('model.pkl' , 'rb') as f:
        lr = pickle.load(f)

    sample=lr.predict(x_val)
    print(sample)

if __name__ == '__main__':
    initial()
