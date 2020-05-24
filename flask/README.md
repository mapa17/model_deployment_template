# Flask Example Deployment
This is an example project that builds an MLflow package and deploys it using flask.

```
python build.py mlflow
```

```
python deploy.pt mlflow
```

## Test deployment

```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"data": [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8]]}' \
  http://localhost:5000/predict
```

## Use gunicorn
```
gunicorn --workers=2 -b localhost:5000 "deploy:_deploy('mlflow')"
```

## Test performance
Use Apache Bench to test the inference performance
```
ab -n 1000 -c 10 -p tests/request.json -s 180 -T application/json -l http://127.0.0.1:5000/predict
```