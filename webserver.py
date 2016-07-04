from flask import request,Flask, url_for

app = Flask(__name__)

from predict import predict, updateWeight

FNAME = "tmp.png"

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print "PROCESSING CLASSIFICATION REQUEST"
        f = request.files['img']
        f.save(FNAME)
        return str(predict(FNAME))

@app.route('/')
def index():
    return 'Index Page'

@app.route('/update')
def update():
    updateWeight()
    return 'updated'

if __name__ == "__main__":
    app.run(host = '0.0.0.0')
