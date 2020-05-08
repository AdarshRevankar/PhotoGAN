import os
import test
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)


@app.route("/")
def view_index_template():
    return render_template('index.html')


@app.route('/gan', methods=['POST'])
def generate():
    # Some of the Parameters
    # Get the Request & Parse it
    rf_keys = request.form.keys()

    # Assign key -> data and Load the Request Dictionary
    data = None
    for key in rf_keys:
        data = key

    # If no data found simply return
    if data is None:
        raise Exception('REQUESTED DATA CANNOT BE NONE')

    data_dic = json.loads(data)

    # Execute the command
    print('\n', "*****" * 20, "\nStarted Executing TEST\n", "*****" * 20)
    test.pre_process_semantic_input()
    test.generate_from_data()
    test.clear_images()
    print('\n', "*****" * 20, "\n  Ended Executing TEST\n", "*****" * 20)

    # Check if the output has obtained
    msg = 'successful' if os.path.exists(output_dir + save_dir + data_dic["filename"]) else 'un-successful'

    # Create the response dictionary
    resp_dic = {
        'img': save_dir + data_dic["filename"],
        'msg': msg
    }

    # Convert to JSON Object
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'

    # Return the JSON response
    print('Generation of Image was "', msg, '"')
    return resp


if __name__ == '__main__':
    save_dir = 'coco_pretrained/test_latest/synthesized_image/'
    output_dir = 'static/output/'
    app.run()
