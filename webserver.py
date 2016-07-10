from flask import request,Flask, url_for

app = Flask(__name__)

from eval import predict, updateWeight

FNAME = "tmp.wav"

@app.route('/upload', methods=['GET', 'POST', 'OPTIONS'])
def upload_file():
    if request.method == 'POST':
        print "PROCESSING CLASSIFICATION REQUEST"
        f = request.files['wav']
        f.save(FNAME)
        return str(predict(FNAME))

@app.route('/')
def index():
    return 'Index Page'

@app.route('/update')
def update():

    try:
        wid = int(request.args.get('q'))
        num = updateWeight(wid)
    except:

        num = updateWeight()
    return "updated to %d" % num

if __name__ == "__main__":
    app.run(host = '0.0.0.0')
