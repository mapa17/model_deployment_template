import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn
import mlflow
import mlflow.pyfunc
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import shutil
import click
from pudb import set_trace as st
import tempfile

# Iris dataset from
# https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv
iris_df = pd.read_csv('iris.csv')


def load_data(data_path):
    df = pd.read_csv(data_path)

    label_encoder = preprocessing.LabelEncoder()

    # Convert labels to ints
    df['variety'] = label_encoder.fit_transform(df['variety'])

    return df, label_encoder


def prepare_data(df):
    # Drop labels from out training features
    X = df.drop('variety', axis = 1)
    Y = df[['variety']]

    # Generate train / test splits for training and evaluation
    xT, xt, yT, yt = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Convert to tensors (make sure y is a single vector)
    xT = torch.from_numpy(xT.values).float()
    xt = torch.from_numpy(xt.values).float()
    yT = torch.from_numpy(yT.values).view(1,-1)[0]
    yt = torch.from_numpy(yt.values).view(1,-1)[0]

    return xT, xt, yT, yt


def prepare_model(model_path : str):
    # Expect model to be a string of the kind
    # model.submodule.IrisNet
    print(f'Loading model {model_path} ...')
    module, class_name = '.'.join(model_path.split('.')[0:-1]), model_path.split('.')[-1]
    mod = __import__(module, fromlist=[class_name])
    model_class = getattr(mod, class_name)

    print(f'Preparing model {model_class} ...')
    # Initialize the network
    model = model_class()

    # Set the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr = 0.03)
    loss_fn = nn.NLLLoss()

    # Add optimizer and loss function to the model object
    model.optimizer = optimizer
    model.loss = loss_fn

    return model


def train_model(model, data, epochs=500):
    xT, xt, yT, yt = data

    print(f'Start training model for {epochs} epochs ...')
    for epoch in range(epochs):
        model.optimizer.zero_grad()
        y_pred = model(xT)
        loss = model.loss(y_pred , yT)
        loss.backward()
        model.optimizer.step()

        if epoch % (epochs // 10) == 0:
            print(f'Epoch: {epoch} loss: {loss.item()}')


# This will serve as an MLflow wrapper for the model
class ModelWrapper(mlflow.pyfunc.PythonModel):
    # Load in the model and all required artifacts
    # The context object is provided by the MLflow framework
    # It will contain all of the artifacts specified above
    def load_context(self, context):
        import torch
        import pickle

        # Load package meta
        with open(context.artifacts["meta"], 'rb') as handle:
            meta = pickle.load(handle)

        # Load the model
        self.model = prepare_model(meta['model_path'])
        
        # Initialize the model and load in the state dict
        self.model.load_state_dict(torch.load(context.artifacts["state_dict"]))

        # Load in and deserialize the label encoder object
        with open(context.artifacts["label_encoder"], 'rb') as handle:
            self.label_encoder = pickle.load(handle)

    # Create a predict function for our models
    def predict(self, context, model_input):
        with torch.no_grad():
            example = torch.tensor(model_input.values)
            pred = torch.argmax(self.model(example.float()), dim=1)
            pred_labels = self.label_encoder.inverse_transform(pred)
        return pred_labels



def pack_model(model, model_path, meta_data, label_encoder, output_path):
    # MLFlow is very picky about the output directory
    # Generate one temporal folder containing artifacts while generating the
    # mlflow package
    print(f'Writing packaged to {output_path} ...')
    shutil.rmtree(output_path, ignore_errors=True)
    #os.makedirs(f'{output_path}/tmp/', exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Serialize the label encoder
        # This will be required at inference time
        #le_path = f'{output_path}/tmp/label_encoder.pkl'
        le_path = f'{tmp}/label_encoder.pkl'
        with open(le_path, 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Build package meta dict
        package_meta = {
            'model_path': model_path,
            'user_meta': meta_data,
            }

        # Meta data about this package
        #meta_path = f'{output_path}/tmp/meta.pkl'
        meta_path = f'{tmp}/meta.pkl'
        with open(meta_path, 'wb') as handle:
            pickle.dump(package_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


        # Serialize the models state_dict
        #state_dict_path = f'{output_path}/tmp/state_dict.pt'
        state_dict_path = f'{tmp}/state_dict.pt'
        torch.save(model.state_dict(), state_dict_path)

        # Here we will create an artifacts object
        # It will contain all of the data artifacts that we want to package with the model
        artifacts = {
            "state_dict": state_dict_path,
            "label_encoder": le_path,
            "meta": meta_path 
        }

        env = mlflow.pyfunc.get_default_conda_env() 

        # Package the model!
        mlflow.pyfunc.save_model(path=f'{output_path}',
                            python_model=ModelWrapper(),
                            artifacts=artifacts,
                            conda_env=env,
                            code_path=[f"{model_path.split('.')[0]}.py", meta_data])
        


def _build(modelpath, data, package, user_meta):
    print(f'Loading data ...')
    df, label_encoder = load_data(data)

    data = prepare_data(df)

    print(f'Preparing model ...')
    model = prepare_model(modelpath)

    print(f'Training ...')
    train_model(model, data)

    print(f'Packing ...')
    pack_model(model, modelpath, user_meta, label_encoder, package)



@click.command()
@click.argument('package')
@click.option('--modelpath', default='model.IrisNet', help='Specify Model')
@click.option('--data', default='iris.csv', help='Path to the training data file')
@click.option('--user_meta', default='meta.json', help='Path to the meta file')
def build(modelpath, data, package, user_meta):
    _build(modelpath, data, package, user_meta)


if __name__ == '__main__':
    build()