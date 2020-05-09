import base64
import os
import re
import test
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def view_index_template():
    return render_template('index.html')


def save_base64_image(image_b64, folder, filename):
    # Converts the Base64 to Binary image data
    imgstr = re.search(r'data:image/png;base64,(.*)', image_b64).group(1)

    # Save the converted binary data
    output = open(os.path.join(folder, filename), 'wb')
    decoded = base64.b64decode(imgstr)
    output.write(decoded)
    output.close()


@app.route('/gan', methods=['POST'])
def generate():
    save_dir = 'coco_pretrained/test_latest/synthesized_image/'
    output_dir = 'static/output/'

    image_b64 = request.values['imageBase64']
    filename = request.values['filename']
    style_index = int(request.values['style_image'])
    drawing_folder = 'datasets/coco_stuff/val_drawing'

    save_base64_image(image_b64, drawing_folder, filename)

    # Execute the command
    print('\n', "*****" * 20, "\nStarted Executing TEST\n", "*****" * 20)
    test.clear_images(exclude_file= filename, max_img_buffer=4)
    test.pre_process_semantic_input(style_index=style_index)
    test.generate_from_data()
    print('\n', "*****" * 20, "\n  Ended Executing TEST\n", "*****" * 20)

    # Check if the output has obtained
    msg = 'successful' if os.path.exists(output_dir + save_dir + filename) else 'un-successful'

    # Create the response dictionary
    resp_dic = {
        'img': save_dir + filename,
        'msg': msg
    }

    # Convert to JSON Object
    resp = jsonify(resp_dic)
    resp.headers['Access-Control-Allow-Origin'] = '*'

    # Return the JSON response
    print('Generation of Image was "', msg, '"')
    return resp


if __name__ == '__main__':
    app.run()
