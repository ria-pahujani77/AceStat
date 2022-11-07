import flask
import pickle
import pandas as pd

with open(f'C:\\Users\\asus\\Desktop\\AceStat\\venv\\webapp\\model\\model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        data1 = flask.request.form['Data1']
        data2 = flask.request.form['Data2']
        data3 = flask.request.form['Data3']
        data4 = flask.request.form['Data4']
        data5 = flask.request.form['Data5']
        data6 = flask.request.form['Data6']
        data7 = flask.request.form['Data7']
        data8 = flask.request.form['Data8']
        data9 = flask.request.form['Data9']
        data10 = flask.request.form['Data10']
        data11 = flask.request.form['Data11']
        data12 = flask.request.form['Data12']
        data13 = flask.request.form['Data13']
        data14 = flask.request.form['Data14']
        data15 = flask.request.form['Data15']
        data16 = flask.request.form['Data16']

        input_variables = pd.DataFrame([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16]],
                                       columns=['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', data11, data12, data13, data14, data15, data16],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'data1':data1,
                                                     'data2':data2,
                                                     'data3':data3,
                                                     'data4':data4,
                                                     'data5':data5,
                                                     'data6':data6,
                                                     'data7':data7,
                                                     'data8':data8,
                                                     'data9':data9,
                                                     'data10':data10,
                                                     'data11':data11,
                                                     'data12':data12,
                                                     'data13':data13,
                                                     'data14':data14,
                                                     'data15':data15,
                                                     'data16':data16},
                                     result=prediction,
                                     )
if __name__ == '__main__':
    app.run()