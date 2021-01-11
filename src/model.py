"""
Class to train contrastive encoder in Self-Supervised setting.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1

A wrapper model to be used in the context of Contrastive Predictive Coding framework.
"""

import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch as th
import itertools
from utils.utils import set_seed
from utils.loss_functions import byol_loss
from utils.model_plot import save_loss_plot
from utils.model_utils import BYOL
th.autograd.set_detect_anomaly(True)


class BYOLModel:
    """
    Model: Wrapper model for BYOL class
    Loss function: InfoNCE - https://arxiv.org/pdf/2006.07733v1.pdf
    """

    def __init__(self, options):
        """
        :param dict options: Configuration dictionary.
        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # Set hyper-parameters
        self.set_params()
        # Set paths for results as well as initializing some arrays to collect data during training
        self.set_paths()
        # ------Network---------
        # Instantiate networks
        print("Building models...")
        # Set BYOL model i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_byol()
        # Print out model architecture
        self.get_model_summary()


    def set_byol(self):
        """Initialize the model, sets up the loss, optimizer, device assignment (GPU, or CPU) etc."""
        # Instantiate the model
        self.byol= BYOL(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"byol": self.byol})
        # Assign the model to a device
        self.byol.to(self.device)
        # Initialize parameters of the target network with that of online network so that they are same at the beginning
        self.update_target_network(initialization=True)
        # Reconstruction loss
        self.byol_loss = byol_loss
        # Set optimizer - We will use it only for the parameters of 'online network' + 'predictor'
        self.optimizer_byol = self._adam()
        # Set scheduler (its usage is optional)
        self.set_scheduler()
        # Add items to summary to be used for reporting later
        self.summary.update({"byol_loss": []})


    def update_target_network(self, initialization=False):
        """Updates parameters of Target network as a moving average of the parameters of Online network."""
        # Turn off gradient computation
        with th.no_grad():
            # Go through corresponding params in online and target networks, and update the parameters of the target
            for q_params, z_params in zip(self.byol.online_network.parameters(), self.byol.target_network.parameters()):
                if initialization:
                    # Update target params: z = q
                    update = q_params.data
                    # Turn off gradients for target parameters
                    q_params.requires_grad = False
                else:
                    # Update target params: z = t*z + (1-t)*q
                    update = self.momentum * z_params.data + (1 - self.momentum) * q_params.data
                # Update target params
                z_params.data = update

    def fit(self, data_loader):
        """
        :param IterableDataset data_loader: Pytorch data loader.
        :return: None

        Fits model to the data using contrastive learning.
        """
        # Training dataset
        train_loader = data_loader.train_loader
        # Validation dataset. Note that it uses only one batch of data to check validation loss to save from computation.
        Xval = self.get_validation_batch(data_loader)
        # Loss dictionary: "v": validation -- Suffixes: "_b": batch, "_e": epoch
        self.loss = {"byol_loss_b": [], "byol_loss_e": [], "vloss_e": []}
        # Turn on training mode for each model.
        self.set_mode(mode="training")
        # Compute total number of batches per epoch
        self.total_batches = len(train_loader)
        print(f"Total number of samples / batches in training set: {len(train_loader.dataset)} / {len(train_loader)}")
        # Start joint training of contrastive_encoder, and/or classifier
        for epoch in range(self.options["epochs"]):
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(train_loader), total=self.total_batches, leave=True)
            # Go through batches
            for i, ((xi, xj), _) in self.train_tqdm:
                # Concatenate xi, and xj, and turn it into a tensor
                Xbatch = self.process_batch(xi, xj)
                # Forward pass. Note: Outputs are equivalent to q = concat[qi, qj, dim=0], z = concat[zi, zj, dim=0]
                q, z = self.byol(Xbatch)
                # Compute reconstruction loss
                byol_loss = self.byol_loss(q, z)
                # Get contrastive loss for training per batch
                self.loss["byol_loss_b"].append(byol_loss.item())
                # Update the parameters of online network as well as predictor
                self.update_model(byol_loss, self.optimizer_byol, retain_graph=True)
                # Update the parameters of target network
                self.update_target_network()
                # Clean-up for efficient memory use.
                del byol_loss
                gc.collect()
                # Update log message using epoch and batch numbers
                self.update_log(epoch, i)
            # Record training loss per epoch
            self.loss["byol_loss_e"].append(sum(self.loss["byol_loss_b"][-self.total_batches:-1]) / self.total_batches)
            # Validate every nth epoch. n=1 by default
            _ = self.validate(Xval) if epoch % self.options["nth_epoch"] == 0 else None
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")

    def set_params(self):
        """Sets up parameters needed for training"""
        # Set momentum used when updating Target network
        self.momentum = self.options["tau"]
        # Set learning rate used to update parameters of Online network
        self.lr = self.options["learning_rate"]

    def update_log(self, epoch, batch):
        """Updated the log message displayed during training"""
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch-1}], Batch:[{batch}] loss:{self.loss['byol_loss_b'][-1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch-1}] loss:{self.loss['byol_loss_e'][-1]:.4f}, val loss:{self.loss['vloss_e'][-1]:.4f}"
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        """Sets the mode of the model. If mode==training, the model parameters are expected to be updated."""
        # Change the mode of models, depending on whether we are training them, or using them for evaluation.
        if mode == "training":
            self.byol.online_network.train()
            self.byol.predictor.train()
        else:
            self.byol.online_network.eval()
            self.byol.predictor.eval()

    def process_batch(self, xi, xj):
        """Concatenates two transformed inputs into one, and moves the data to the device as tensor"""
        # Combine xi and xj into a single batch
        Xbatch = np.concatenate((xi, xj), axis=0)
        # Convert the batch to tensor and move it to where the model is
        Xbatch = self._tensor(Xbatch)
        # Return batches
        return Xbatch

    def get_validation_batch(self, data_loader):
        """Wrapper to get validation set. In this case, it uses only the first batch to save from computation time"""
        # Validation dataset
        validation_loader = data_loader.test_loader
        # Use only the first batch of validation set to save from computation
        ((xi, xj), _) = next(iter(validation_loader))
        # Concatenate xi, and xj, and turn it into a tensor
        Xval = self.process_batch(xi, xj)
        # Return
        return Xval

    def validate(self, Xval):
        """Computes validation loss"""
        with th.no_grad():
            # Turn on evaluation mode
            self.set_mode(mode="evaluation")
            # Forward pass on contrastive_encoder
            q, z = self.byol(Xval)
            # Compute reconstruction loss
            byol_vloss = byol_loss(q, z)
            # Get contrastive loss for training per batch
            self.loss["vloss_e"].append(byol_vloss.item())
            # Turn on training mode
            self.set_mode(mode="training")
            # Clean up to avoid memory issues
            del byol_vloss, q, z, Xval
            gc.collect()

    def save_weights(self):
        """
        :return: None
        Used to save weights of contrastive_encoder, and (if options['supervision'] == 'supervised) Classifier
        """
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + ".pt")
        print("Done with saving models.")

    def load_models(self):
        """
        :return: None
        Used to load weights saved at the end of the training.
        """
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + ".pt")
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def get_model_summary(self):
        """
        :return: None
        Sanity check to see if the models are constructed correctly.
        """
        # Summary of contrastive_encoder
        description  = f"{40*'-'}Summarize models:{40*'-'}\n"
        description += f"{34*'='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34*'='}\n"
        description += f"{self.byol}\n"
        # Print model architecture
        print(description)

    def update_model(self, loss, optimizer, retain_graph=True):
        """
        :param loss: Loss to be used to compute gradients
        :param optimizer: Optimizer to update weights
        :param retain_graph: If True, keeps computation graph
        :return:
        """
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def set_scheduler(self):
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_byol, step_size=2, gamma=0.99)

    def set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = self.options["paths"]["results"]
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", self.options["model_mode"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", self.options["model_mode"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", self.options["model_mode"], "loss")

    def _adam(self):
        """Wrapper for setting up Adam optimizer"""
        # Collect params
        params = [self.byol.online_network.parameters(), self.byol.predictor.parameters()]
        # Return optimizer
        return th.optim.Adam(itertools.chain(*params), lr=self.lr, betas=(0.9, 0.999))

    def _tensor(self, data):
        """Wrapper for moving numpy arrays to the device as a tensor"""
        return th.from_numpy(data).to(self.device).float()
