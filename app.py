from flask import Flask, request, jsonify
from scripts.models import CRFModel
from scripts.transformer import Transformer
import config


app = Flask(__name__)
crf_model = CRFModel()
crf_model.load(config.model_root + 'crf_suite_v1')
tf = Transformer()


@app.route('/')
def index():
    return ('/init, /evaluate, /fit, /predict')


@app.route('/predict')
def predict():
    text = request.args.get('text')
    features = tf.convertTextToNgram(text)
    pred_labels = crf_model.predict(features)
    print(pred_labels)
    ann_data = tf.convertLabelsToAnn(text, pred_labels)
    return jsonify(ann_data)


if __name__ == "__main__":
    app.run(host='localhost')
