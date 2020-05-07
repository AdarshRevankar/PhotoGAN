from subprocess import call
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

    # Execute the command
    print("*****" * 20, "\nStarted Executing TEST\n", "*****" * 20)
    call(["python", "test.py", "--style_index", str(int(data_dic["style_image"]) + 1)])
    print("*****" * 20, "\n  Ended Executing TEST\n", "*****" * 20)

    resp_dic = {
        'img': "coco_pretrained/test_latest/synthesized_image/Grassland.png",
        'msg': 'successful'
    }
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


# run the application
if __name__ == "__main__":
    app.run(debug=True)
