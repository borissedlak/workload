from datetime import datetime

from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Save the file
    file.save(f'/home/boris/data/metrics_{datetime.now()}.csv')
    print("Performance file uploaded")

    return 'Performance file uploaded'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
