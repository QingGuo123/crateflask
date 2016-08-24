# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template
from crate.client import connect
import xgboost as xgb
import json
import datetime
import numpy as np
from urllib.parse import unquote
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def get_geometry_data():
    try:
        search = unquote(unquote(request.values.get('search')))
        search_json = json.loads(search)

        polygon = search_json['coordinates']
        
        # particleSize = search_json['particleSize']
        # interval = datetime.timedelta(minutes=int(particleSize))

        coordinates = [list(map(lambda x: [x['lng'], x['lat']], polygon))]
        if coordinates[0][0] != coordinates[0][-1]:
            coordinates[0].append(coordinates[0][0])

        connection = connect(['100.69.123.31:8009', '100.69.122.50:8009', '100.69.122.15:8009'])
        cursor = connection.cursor()
        print(coordinates)

        final_return = get_predict(cursor, coordinates)
    finally:
        cursor.close()
        connection.close()

    
    return final_return


def get_predict(cursor, coordinates):

    
    #fetch feature data from crate.io
    cursor.execute("""
                select ROUND(birth_time/900000) AS round,(extract(DAY_OF_MONTH FROM birth_time) ) AS day_of_month,
                (extract(DAY_OF_WEEK FROM birth_time) ) AS day_of_week,sum(start_dest_distance),sum(order_status/7)as cancel,count(*) as cnt
                from binlog.gulfstream
                where within(starting_posi,{TYPE = 'Polygon',coordinates = %s })
                group by round,day_of_month,day_of_week
                order by round 
                """ % (coordinates))
  

    all_data = np.array(cursor.fetchall())
    alldata = np.vsplit(all_data,[all_data.shape[0]-1,all_data.shape[0]])

    split_data = np.hsplit(alldata[0], [5,6])


    prelabel = split_data[1]

    second_split = np.hsplit(split_data[0],[1,5])
    prefeature = second_split[1]


    last_time = second_split[0]
    print(last_time)
    count = last_time.shape[1]
    first_time = (last_time[count].astype(int))
    
  
    time = []
    for i in range(1, 10+1):
        temp_time = datetime.datetime.fromtimestamp((first_time+i)*900).strftime('%H:%M')
        time.append(temp_time)
       


    print(prefeature)
    print(prelabel)
    print(time)
    timeseries = np.array(time).tolist()

    # time = split_data[0]
    # last_time = time[time.shape[0]].astype(int).tolist()

    feature_split = np.vsplit(prefeature,[prefeature.shape[0]-10,prefeature.shape[0]])
    feature_train = feature_split[0]
    #print(feature_split[1])
    #print(np.zeros((10,5)))

    feature_test = np.row_stack((feature_split[1],np.zeros((10,4))))

    label_split = np.vsplit(prelabel,[9,10])
    label_train = label_split[2]

    #print(label_split[0])

    #XGboost
  
    dtrain = xgb.DMatrix(feature_train, label=label_train)
    dtest = xgb.DMatrix(feature_split[1])
    param = {'booster':'gbtree','max_depth':15, 'eta':0.5,'silent':0, 'objective':'reg:linear' }
    evallist  = [(dtrain,'train')]
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist )
    xgdpred = bst.predict(dtest)
    xgdpredict = xgdpred.astype(int).tolist()
    print (xgdpredict)

    #RandomForest
    rf = RandomForestRegressor(n_estimators = 50,max_depth=18)
    rf.fit(feature_train,label_train)
    rfpred = rf.predict(feature_split[1])
    rfpredict = rfpred.astype(int).tolist()
    print (rfpredict)

    #SGD
    sgd = linear_model.SGDRegressor()
    sgd.fit(feature_train,label_train)
    sgdpred = sgd.predict(feature_split[1])
    sgdpredict = sgdpred.astype(int).tolist()
    
    print (sgdpredict)


    #return np.row_stack((xgpred.astype(int),rfpred.astype(int),sgdpred.astype(int)))
    return json.dumps({"xgboost":xgdpredict,"randomforest":rfpredict,"sgd":xgdpredict,"time":timeseries})


def generate_day_xAxis(particleSizedelta):
    date = '2016-08-01'
    start_datetime_type = datetime.datetime.strptime(date, '%Y-%m-%d')
    stop_datetime_type = start_datetime_type + datetime.timedelta(days=1)

    current_time = start_datetime_type
    final_list = []
    while current_time < stop_datetime_type:
        final_list.append(current_time)
        current_time = current_time + particleSizedelta

    return list(map(lambda date: str(date)[11:16], final_list))


def toTimeStamp(datetimeobj):
    return str(int(datetimeobj.timestamp())) + "000"


def list2map(l):
    return {"lng": l[0], "lat": l[1], "count": l[2]}


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8002)