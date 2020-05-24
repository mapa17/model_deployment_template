import pytest
import shutil
import os.path
import requests

from deploy import _deploy
from build import _build

@pytest.fixture
def base_model(scope='module'):
    output_path = 'tests/base_model'
    shutil.rmtree(output_path, ignore_errors=True) 
    _build('model.IrisNet', 'iris.csv', output_path, 'meta.json')
    yield output_path
    shutil.rmtree(output_path, ignore_errors=True) 


@pytest.fixture
def deployment(base_model):
    app = _deploy(base_model)
    with app.test_client() as client:
        yield client


def test_building(base_model):
    assert os.path.exists(base_model)


def test_prediction(deployment):
    req = {'data': [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8]]}
    resp = deployment.post('/predict', json=req)
    assert resp.data.strip() ==  b'["Setosa","Virginica"]'
