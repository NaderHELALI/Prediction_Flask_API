from flask import Flask, render_template,request
import models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def Processing():
    data = pd.read_csv('online_shoppers_intention.csv')
    data['Weekend'] = data['Weekend'].map({False: 0, True: 1})
    data['Revenue'] = data['Revenue'].map({False: 0, True: 1})
    y = data.Revenue
    X = data.drop("Revenue", axis=1)
    X = pd.get_dummies(X, columns=['VisitorType'], drop_first=True)
    X = pd.get_dummies(X, columns=['Month'], drop_first=True)
    return [X, y]

@app.route("/")
def home():
    return render_template('home.html')


@app.route('/randomForest', methods=['GET', 'POST'])
def randomForest():
    data = Processing()
    X_train, X_test, y_train, y_test = train_test_split(data[0],
                                                        data[1],
                                                        test_size=0.3,
                                                        random_state=2)
    model = models.Random_forest_model(X_train, y_train)
    if request.method == 'GET':
        return (render_template('random_forest.html'))
    if request.method == 'POST':

        Administrative = request.form['Administrative']
        Administrative_Duration = request.form['Administrative_Duration']
        Informational = request.form['Informational']
        Informational_Duration = request.form['Informational_Duration']
        ProductRelated = request.form['ProductRelated']
        ProductRelated_Duration = request.form['ProductRelated_Duration']
        BounceRates = request.form['BounceRates']
        ExitRates = request.form['ExitRates']
        PageValues = request.form['PageValues']
        SpecialDay = request.form['SpecialDay']
        OperatingSystems = request.form['OperatingSystems']
        Browser = request.form['Browser']
        Region = request.form['Region']
        TrafficType = request.form['TrafficType']
        Weekend = request.form['Weekend']
        Weekend_stored=Weekend
        Month = request.form['Month'] 
        VisitorType = request.form['VisitorType']

        Weekend=0 if Weekend else 1

        input_variables = pd.DataFrame([[Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, OperatingSystems, Browser, Region, TrafficType, Weekend, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=['Administrative', 'Administrative_Duration', 'Informational',
                                                                                                                                                                                                                                                                                                                       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                                                                                                                                                                                                                                                                                                                       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                                                                                                                                                                                                                                                                                                                       'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend',
                                                                                                                                                                                                                                                                                                                       'VisitorType_Other', 'VisitorType_Returning_Visitor', 'Month_Dec',
                                                                                                                                                                                                                                                                                                                       'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May',
                                                                                                                                                                                                                                                                                                                       'Month_Nov', 'Month_Oct', 'Month_Sep'])
    
        for temp in input_variables.columns[-9] :
            if Month in temp:
                input_variables[temp][0]=1
        
        for temp in input_variables.columns[15:16] :
            if VisitorType in temp :
                input_variables[temp][0]=1
        
        prediction = bool(models.getPrediction(input_variables, model)[0])
        print(prediction)
        
        return render_template('random_forest.html', original_input={'Administrative': Administrative, 'Administrative_Duration': Administrative_Duration, 'Informational_Duration': Informational_Duration, 'ProductRelated': ProductRelated, 'ProductRelated_Duration': ProductRelated_Duration, 'BounceRates': BounceRates, 'ExitRates': ExitRates, 'PageValues': PageValues, 'SpecialDay': SpecialDay, 'OperatingSystems': OperatingSystems,
                                                                     'Browser': Browser, 'Region': Region, 'TrafficType': TrafficType, 'Weekend': Weekend_stored, 'Month': Month,'VisitorType': VisitorType},
                               result=prediction)
if __name__ == '__main__':
    app.run(debug=True)