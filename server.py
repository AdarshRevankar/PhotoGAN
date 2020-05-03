from flask import Flask, request, jsonify
import json
from flask import render_template

# creates a Flask application, named app
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/gan', methods=['POST'])
def sum_num():
    rf = request.form
    for key in rf.keys():
        data = key
    print("Data Recieved from front-End", data)
    data_dic = json.loads(data)
    resp_dic = {
        'img': "mountain.png",
        'msg': 'successful'
    }
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


# run the application
if __name__ == "__main__":
    app.run(debug=True)
