from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import json
import click
import pickle

## Test in terminal with
"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"data": [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8]]}' \
  http://localhost:5000/predict
"""

# Name of the apps module package
app = Flask(__name__)
flask_env = {
    'model': None,
    'user_meta': None
}


# Meta data endpoint
@app.route('/', methods=['GET'])
def meta_data():
	return jsonify(flask_env['user_meta'])


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
	req = request.get_json()
	
	# Log the request
	print({'request': req})

	# Format the request data in a DataFrame
	inf_df = pd.DataFrame(req['data'])

	# Get model prediction - convert from np to list
	pred = flask_env['model'].predict(inf_df).tolist()

	# Log the prediction
	print({'response': pred})

	# Return prediction as response
	return jsonify(pred)


def _deploy(package):
    # Load in the model at app startup
    model = mlflow.pyfunc.load_model(package)

    # Load package metadata
    with open(model.context.artifacts["meta"], 'rb') as handle:
        meta = pickle.load(handle)

    # Load user metadata
    with open(f"{package}/code/{meta['user_meta']}", 'rb') as handle:
        user_meta = json.load(handle)

    flask_env['model'] = model
    flask_env['meta'] = meta
    flask_env['user_meta'] = user_meta

    return app


@click.command()
@click.argument('package', type=click.Path(exists=True))
@click.option('--port', default=5000, help='Port to serve model')
@click.option('--debug', default=False, help='Enable flask debuggin', is_flag=True)
def deploy(package, port, debug):
    app = _deploy(package)
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    deploy()