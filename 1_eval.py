"""
Evaluation of the model and baseline performance using linear classifier ( Logistic regression).
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import mlflow
from src.model import BYOLModel
from utils.load_data import Loader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.arguments import print_config_summary
from utils.arguments import get_arguments, get_config
from utils.utils import set_dirs



def eval(config):
    """
    :param dict config: Generic dictionary to configure the model for training.
    :return: None
    """
    # Don't use unlabelled data in train loader
    config["unlabelled_data"] = False
    # Pre-process, save, and load data. train=False is used to turn of transformations for training set.
    data_loader = Loader(config, download=True, get_all_train=True, get_all_test=True, train=False)
    # Get data. If check_sample==True, it will show shome images from the training set as a sanity check.
    Xtrain, ytrain, Xtest, ytest, Xtrain2D, Xtest2D = transform_data(data_loader, check_samples=False)
    # Print which dataset we are using
    print(f"{config['dataset']} is being used to test performance.")
    # Get the performance using contrastive encoder
    model_performance(Xtrain, ytrain, Xtest, ytest)
    # Get the baseline performance
    baseline_performance(Xtrain2D, ytrain, Xtest2D, ytest)

def baseline_performance(Xtrain2D, ytrain, Xtest2D, ytest):
    # Turn tensors into numpy arrays since they will be used with PCA
    Xtrain2D, Xtest2D = Xtrain2D.cpu().detach().numpy(), Xtest2D.cpu().detach().numpy()
    # Lower feature dimension to same as the one of the output dimension of contranstive encoder to be a fair comparison
    pca = PCA(n_components=config["conv_dims"][-1])
    # Fit training data and transform
    X_train_pca = pca.fit_transform(Xtrain2D)
    # Transform test data
    X_test_pca = pca.transform(Xtest2D)
    # Baseline performance using PCA
    linear_model_eval(X_train_pca, ytrain, X_test_pca, ytest, description="Baseline Performance")

def model_performance(Xtrain, ytrain, Xtest, ytest):
    # Instantiate model
    model = BYOLModel(config)
    # Load contrastive encoder
    model.load_models()
    # Change the mode to evaluation
    model.set_mode("evaluation")
    # Extract encoder model
    encoder = model.byol.online_network.encoder
    # Get representations using Training and Test sets
    h_train = encoder(Xtrain)
    h_test = encoder(Xtest)
    # Turn tensors into numpy arrays since they will be used with Logistic regression model
    h_train, h_test = h_train.cpu().detach().numpy(), h_test.cpu().detach().numpy()
    # Get performance
    linear_model_eval(h_train, ytrain, h_test, ytest, description="BYOL Performance")

def linear_model_eval(X_train, y_train, X_test, y_test, use_scaler=False, description="Baseline: PCA + Logistic Reg."):
    # If true, scale data using scikit-learn scaler
    X_train,  X_test = scale_data(X_train, X_test) if use_scaler else X_train, X_test
    # Initialize Logistic regression
    clf = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
    # Fit model to the data
    clf.fit(X_train, y_train)
    # Summary of performance
    print(50*"="+description+50*"=")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))

def transform_data(data_loader, check_samples=False):
    # Training dataset
    train_loader = data_loader.train_loader
    # Validation dataset
    test_loader = data_loader.test_loader
    # Show samples from training set
    _ = show_samples(train_loader) if check_samples else None
    # Get training and test data. Iterator returns a tuple of 3 variables. Pick the first ones as Xtrain, and Xtest
    ((Xtrain, _), ytrain) = next(iter(train_loader))
    ((Xtest, _), ytest)  = next(iter(test_loader))
    # Print informative message as a sanity check
    print(f"Number of samples in training set: {Xtrain.shape}")
    # Make it a 2D array of batch_size x remaining dimension so that we can use it with PCA for baseline performance
    Xtrain2D, Xtest2D = Xtrain.view(Xtrain.shape[0], -1), Xtest.view(Xtest.shape[0], -1)
    # Return arrays
    return Xtrain, ytrain, Xtest, ytest,  Xtrain2D, Xtest2D

def scale_data(Xtrain, Xtest):
    # Initialize scaler
    scaler = StandardScaler()
    # Fit and transform representations from training set
    Xtrain = scaler.fit_transform(Xtrain)
    # Transform representations from test set
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest

def show_samples(train_loader):
    # Get training data
    ((Xtrain, _), _) = next(iter(train_loader))
    # Turn tensor into numpy array
    Xtrain = Xtrain.numpy()
    # Initialize figure and show samples from training set
    fig, axs = plt.subplots(nrows=2, ncols=6, constrained_layout=False, figsize=(12, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(Xtrain[i].transpose(1, 2, 0))
    plt.show()

def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Start training
    eval(config)

if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        mlflow.set_experiment(experiment_name=config["model_mode"]+"_"+str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main
            main(config)
    else:
        # Run the main
        main(config)