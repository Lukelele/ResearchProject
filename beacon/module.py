import numpy as np
import torch
from tqdm.auto import tqdm
from torchsummary import summary
from . import metrics as beaconmetrics


class Module(torch.nn.Module):
    """
    A base class for creating neural network modules in PyTorch-Beacon. Inherits from the base PyTorch Module class, and adds additional functionalities.
    
    Attributes:
    -----------
    loss_function: torch.nn.modules.loss._Loss
        The loss function to be used during training.
    optimiser: torch.optim.Optimizer
        The optimizer to be used during training.
    learning_rate: float
        The learning rate to be used during training.
    device: str
        The device to be used for training.
    
    Methods:
    --------
    compile(loss_function=torch.nn.CrossEntropyLoss, optimiser=torch.optim.Adam, learning_rate=0.1, device: str = "cpu", optimisations=False):
        Configures the module for training.
        
    fit(dataloader: torch.utils.data.DataLoader, epochs=10):
        Trains the module on the given data.
    """
    def __init__(self):
        super().__init__()
        
        
    def compile(self, loss_function=torch.nn.CrossEntropyLoss, optimiser=torch.optim.Adam, learning_rate:float=0.1, metrics:list=[], device: str = "cpu", optimisations:bool=False):
        """
        Configures the model for training with the specified parameters.

        Parameters:
        - loss_function (torch.nn.modules.loss._Loss, optional): The loss function to use for training the model. Default is torch.nn.CrossEntropyLoss.
        - optimiser (torch.optim.Optimizer, optional): The optimizer to use for training the model. Default is torch.optim.Adam.
        - learning_rate (float, optional): The learning rate to use for training the model. Default is 0.1.
        - metrics (list, optional): A list of metrics to evaluate during training. Default is an empty list.
        - device (str, optional): The device to use for training the model. Default is "cpu".
        - optimisations (bool, optional): Whether to apply optimizations to the model. Default is False.

        Returns:
        - None
        """
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.metrics = []
        self.device = device

        if (len(metrics) > 0):
            if isinstance(metrics[0], str):
                for metric in metrics:
                    self.metrics.append(beaconmetrics.lookup_table[metric])
            else:
                self.metrics = metrics

        if optimisations:
            self = torch.compile(self)

        self.loss_function = self.loss_function()
        self.optimiser = self.optimiser(self.parameters(), self.learning_rate)
            
            
    def fit(self, dataloader: torch.utils.data.DataLoader, epochs=10, track_metrics=True):
        """
        Trains the model on the specified dataloader for the specified number of epochs.

        Args:
        - dataloader: The dataloader to use for training the model.
        - epochs: The number of epochs to train the model for. Default is 10.

        Returns:
        - None
        """
        self.to(self.device)

        metrics = np.zeros((1+len(self.metrics), epochs))

        self.train()

        for epoch in tqdm(range(epochs)):
            for (x, y) in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                y_pred = self(x)

                loss = self.loss_function(y_pred, y)

                batch_metrics = np.zeros(1+len(self.metrics))
                batch_metrics[0] += loss

                if (track_metrics):
                    for i, metric in enumerate(self.metrics):
                        batch_metrics[i+1] += metric(y_pred, y)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            for i in range(batch_metrics):
                metrics[i][epoch] = (batch_metrics[i] / len(dataloader)).item()

        return metrics



    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluates the model on the specified dataloader.

        Args:
        - dataloader: The dataloader to use for evaluating the model.

        Returns:
        """
        self.to(self.device)
        self.eval()

        metrics = np.zeros(1+len(self.metrics))

        with torch.inference_mode():
            loss = 0
            for (x, y) in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                metrics[0] += self.loss_function(y_pred, y)

                for i, metric in enumerate(self.metrics):
                    metrics[i+1] += metric(y_pred, y)

            for i in range(metrics):
                metrics[i] = metrics[i] / len(dataloader)


        return loss.item()


    def fit_tensor(self, X: torch.Tensor, y: torch.Tensor, epochs=10, track_metrics=True):
        """
        Trains the model on the specified input and target tensors for the specified number of epochs.

        Args:
        - X: The input tensor to use for training the model.
        - y: The target tensor to use for training the model.
        - epochs: The number of epochs to train the model for. Default is 10.

        Returns:
        - None
        """
        self.to(self.device)
        X, y = X.to(self.device), y.to(self.device)

        metrics = np.zeros((1+len(self.metrics), epochs))

        self.train()

        for epoch in tqdm(range(epochs)):
            self.optimiser.zero_grad()
            
            y_pred = self(X)
            loss = self.loss_function(y_pred, y)

            metrics[0][epoch] = loss

            if (track_metrics):
                for i, metric in enumerate(self.metrics):
                    metrics[i+1][epoch] = metric(y_pred, y)

            loss.backward()
            self.optimiser.step()

        return metrics
    
    
    def evaluate_tensor(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluates the model on the specified input and target tensors.

        Args:
        - X: The input tensor to use for evaluating the model.
        - y: The target tensor to use for evaluating the model.

        Returns:
        - Tuple of loss.
        """
        self.to(self.device)
        X, y = X.to(self.device), y.to(self.device)
        self.eval()

        metrics = np.zeros(1+len(self.metrics))

        with torch.inference_mode():
            y_pred = self(X)
            metrics[0] = self.loss_function(y_pred, y)

            for i, metric in enumerate(self.metrics):
                    metrics[i+1] = metric(y_pred, y)

        return metrics
    
    
    def predict(self, inputs: torch.Tensor):
        """
        Predicts the output of the model for the given input tensor.

        Args:
        - inputs: The input tensor for which to predict the output.

        Returns:
        - The predicted output tensor.
        """
        self.to(self.device)
        self.eval()
        
        with torch.inference_mode():
            y_pred = self(inputs.to(self.device))
            
        return y_pred
    

    def summary(self, input_size, verbose=False):
        """
        Prints the summary of the model architecture.

        Returns:
        - The summary of the model architecture as a string.
        """
        return summary(self, input_size, verbose=verbose)
    
    
    def save(self, filepath: str):
        """
        Saves the state dictionary of the model to the specified file path.

        Args:
        - filepath: The file path to save the model state dictionary to. If the file path does not end with ".pt" or ".pth", ".pt" will be appended to the file path.

        Returns:
        - None
        """
        if filepath.endswith(".pt") or filepath.endswith(".pth"):
            torch.save(self.state_dict(), filepath)
        else:
            torch.save(self.state_dict(), filepath + ".pt")
            
    
    def load(self, filepath: str):
        """
        Loads the state dictionary of the model from the specified file path.

        Args:
        - filepath: The file path to load the model state dictionary from.

        Returns:
        - None
        """
        self.load_state_dict(torch.load(filepath))



class Sequential(Module):
    """
    A sequential container for holding a sequence of modules.
    Modules will be added to the container in the order they are passed as arguments.
    The forward method will call each module in the sequence in the order they were added.
    
    Args:
        *args: Variable length argument list of modules to be added to the container.
    """
    def __init__(self, *args):
        super().__init__()
        
        for i, module in enumerate(args):
            self.add_module(str(i), module)
        
                
    def forward(self, inputs):
        for module in self:
            inputs = module(inputs)
        return inputs
    
    
    def __iter__(self):
        return iter(self._modules.values())
    