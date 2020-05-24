import pandas as pd
import mlflow.pytorch

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model('output/mlflow')

# Evaluate the model
test_predictions = loaded_model.predict(pd.DataFrame([[5.1, 3.5, 1.4, 0.2]]))

print(test_predictions)