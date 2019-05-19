from flask import Flask, request, render_template
import pickle, os, sys
from logistic_regression import LogisticRegression

app = Flask(__name__, template_folder='html')

def load_model(model_path):
    model_time = os.path.getmtime(model_path)
    model = pickle.load(open(model_path, 'rb'))
    print('>>>> model loaded\t' + str(model_time))
    return model, model_time

def check_and_update_model(model_path):
    global model, loaded_model_time
    # Compare modified times of currently loaded one and the one saved lately
    if os.path.getmtime(model_path) > loaded_model_time:
        model, loaded_model_time = load_model(model_path)

@app.route('/')
def index():
    return render_template('cvr.html')

@app.route('/get/cvr', methods = ['GET'])
def get_cvr():
    check_and_update_model(model_path)
    line = request.args.get('lines')
    return str(model.test_single(line))

@app.route('/update/model')
def update_model():
    global model, loaded_model_time
    model, loaded_model_time = load_model(model_path)
    return 'Model reloaded'

if __name__ == "__main__":
    """
    Should run this program passing one argument:
    python server.py <model file path(where the model should be saved)>
    ex) python server.py ./model/model.dat

    Open a web browser and connect to 
    http://localhost:8080
    """
    if len(sys.argv) != 2:
        print('Invalid number of arguments')
        exit()
    model_path = sys.argv[1] # './model/model.dat'
    model, loaded_model_time = None, None
    model, loaded_model_time = load_model(model_path)
    app.run(port='8080')

    