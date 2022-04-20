from typing import Tuple, Optional

from torch import manual_seed, Tensor, device as torch_device, cuda, no_grad, argmax
from torch.nn import CrossEntropyLoss, Sequential, Module
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Normalize

from arf.conf.env import RESNET_BATCH_SIZE, TRAINING_DATA, VALIDATION_DATA, TEST_DATA, RESNET_BLOCKS, RESNET_EPOCHS, \
    RESNET_IMAGE_SIZE
from arf.constants import ONE_HOT_LABEL
from arf.data import XRayChestDataset
from arf.models import ResNet

DEVICE = torch_device('cuda' if cuda.is_available() else 'cpu')

manual_seed(17)


def load_datasets(batch_size: int = 1, dimensions: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """This function loads the datasets for training, validation and testing.

    Parameters
    ----------
    batch_size : int, optional
        The batch size.
    dimensions : int, optional
        The dimensions of the images.

    Returns
    -------
    `DataLoader` for training, validation and testing.
    """
    
    transformation = Sequential(
        Normalize([0., 0., 0.], [255., 255., 255.]),
        Resize([dimensions, dimensions])
    )
    training = XRayChestDataset(
        TRAINING_DATA, label_type=ONE_HOT_LABEL, images_transformations=transformation, init=True, channels_first=True
    )
    validation = XRayChestDataset(
        VALIDATION_DATA, label_type=ONE_HOT_LABEL, images_transformations=transformation, init=True, channels_first=True
    )
    test = XRayChestDataset(
        TEST_DATA, label_type=ONE_HOT_LABEL, images_transformations=transformation, init=True, channels_first=True
    )
    
    return (
        DataLoader(training, shuffle=True, batch_size=batch_size),
        DataLoader(validation, shuffle=True, batch_size=batch_size),
        DataLoader(test, shuffle=True, batch_size=batch_size)
    )


def training_status(epoch: int, batch: int, loss: float, accuracy: float, train_len: int, *,
                    val_loss: Optional[float] = None,
                    val_acc: Optional[float] = None,
                    total_epochs: Optional[int] = None) -> None:
    """This function prints the status of the training.
    
    Parameters
    ----------
    epoch : int
        The current epoch.
    batch : int
        The current batch.
    loss : float
        The current loss.
    accuracy : float
        The current accuracy.
    train_len : int
        The length of the training dataset divided by batch size.
    val_loss : float, optional
        The current validation loss.
    val_acc : float, optional
        The current validation accuracy.
    total_epochs : int, optional
        The total number of epochs.
    """
    message = f'Epoch {epoch:5d}/{total_epochs or 0}: ' \
              f'{batch:5d}' \
              f' / {train_len:5d} ' \
              f'loss: {loss:.3f} ' \
              f'accuracy {accuracy:.3f}'
    end = '\r'
    if val_loss is not None or val_acc is not None:
        message = f'{message} val_loss: {val_loss:.3f} val_accuracy: {val_acc:.3f}'
        end = '\n'
    print(message, end=end)


def accuracy(output: Tensor, target: Tensor) -> float:
    """This function calculates the accuracy of the model.
    
    Parameters
    ----------
    output : torch.Tensor
        The output of the model.
    target : torch.Tensor
        The target of the model.
    
    Returns
    -------
    float
        The accuracy of the model.
    """
    return (argmax(output, 1) == argmax(target, 1)).sum().item() / float(target.shape[0])


def epoch_loop(
        dataset: DataLoader,
        model: Module,
        optimizer: Optimizer,
        criterion: _Loss,
        epoch: int,
        total_batches: int,
        show_progress: bool = True,
        *,
        total_epochs: Optional[int] = None
) -> Tuple[float, float]:
    """This function trains the model for one epoch.
    
    Parameters
    ----------
    dataset : torch.utils.data.DataLoader
        The dataset for training.
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    criterion : torch.nn.modules.loss._Loss
        The loss function to use.
    epoch : int
        The current epoch.
    total_batches : int
        The total number of batches.
    show_progress : bool = True
        Whether to show the progress of the training.
    total_epochs : int, optional
        The total number of epochs.
    """
    running_loss = 0.0
    acc = 0.0
    for i, (inputs, labels) in enumerate(dataset):
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        inputs = inputs.to(DEVICE)
        labels = labels.float()
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs.to(DEVICE)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        acc += accuracy(outputs, labels)
        if show_progress:
            training_status(epoch, i, running_loss / (i + 1), acc / (i + 1), total_batches, total_epochs=total_epochs)
    return running_loss, acc


def evaluate(dataset: DataLoader, model: Module, criterion: _Loss):
    """This function evaluates the model. It returns the loss and accuracy.
    
    Parameters
    ----------
    dataset : torch.utils.data.DataLoader
        The dataset for evaluation.
    model : torch.nn.Module
        The model to evaluate.
    criterion : torch.nn.modules.loss._Loss
        The loss function to use.
    
    Returns
    -------
    float : The loss of the model.
    float : The accuracy of the model.
    """
    running_loss = 0.0
    acc = 0.0
    with no_grad():
        for i, (inputs, labels) in enumerate(dataset):
            inputs = inputs.to(DEVICE)
            labels = labels.float()
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            acc += accuracy(outputs, labels)
    return running_loss, acc


def train_model(epochs: int, train: DataLoader, model: Module, optimizer: Optimizer, criterion: _Loss, *,
                validation: Optional[DataLoader] = None, total_epochs: Optional[int] = None) -> None:
    """This function trains the model. This functions shows the progress of the
    training about loss and accuracy. If a validation dataset is provided, the
    validation loss and accuracy will be shown at the end of each epoch.
    
    Parameters
    ----------
    epochs : int
        The number of epochs to train for.
    train : torch.utils.data.DataLoader
        The training dataset.
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    criterion : torch.nn.modules.loss._Loss
        The loss function to use.
    validation : torch.utils.data.DataLoader, optional
        The validation dataset.
    total_epochs : int, optional
        The total number of epochs.
    """
    cuda.empty_cache()
    model = model.to(DEVICE)
    
    total_batches = len(train)
    total_val_batches = len(validation)
    conf = model, optimizer, criterion
    for epoch in range(1, epochs + 1):
        epoch_conf = *conf, epoch, total_batches
        running_loss, acc = epoch_loop(train, *epoch_conf, show_progress=True, total_epochs=total_epochs)
        
        if validation is None:
            continue
        val_running_loss, val_acc = evaluate(validation, model, criterion)
        training_status(
            epoch,
            total_batches,
            running_loss / total_batches,
            acc / total_batches,
            total_batches,
            val_loss=val_running_loss / total_val_batches,
            val_acc=val_acc / total_val_batches,
            total_epochs=total_epochs
        )


def train_resnet():
    """This function trains the ResNet model."""
    train, val, test = load_datasets(RESNET_BATCH_SIZE, RESNET_IMAGE_SIZE)
    criterion = CrossEntropyLoss()
    model = ResNet(RESNET_BLOCKS, num_classes=2, input_dimensions=RESNET_IMAGE_SIZE)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    train_model(RESNET_EPOCHS, train, model, optimizer, criterion, validation=test, total_epochs=RESNET_EPOCHS)
