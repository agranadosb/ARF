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


def load_datasets() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """This function loads the datasets for training, validation and testing.
    
    Returns
    -------
    `DataLoader` for training, validation and testing.
    """
    
    transformation = Sequential(
        Normalize([0., 0., 0.], [255., 255., 255.]),
        Resize([RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE])
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
        DataLoader(training, shuffle=True, batch_size=RESNET_BATCH_SIZE),
        DataLoader(validation, shuffle=True, batch_size=RESNET_BATCH_SIZE),
        DataLoader(test, shuffle=True, batch_size=RESNET_BATCH_SIZE)
    )


def training_status(epoch: int, batch: int, loss: float, accuracy: float, train_len: int,
                    val_loss: Optional[float] = None,
                    val_acc: Optional[float] = None) -> None:
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
    """
    message = f'Epoch {epoch}/100: ' \
              f'{batch:5d}' \
              f' / {train_len} ' \
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
            training_status(epoch, i, running_loss / (i + 1), acc / (i + 1), total_batches)
    return running_loss, acc


def train_resnet():
    """This function trains the ResNet model."""
    train, val, test = load_datasets()
    criterion = CrossEntropyLoss()
    model = ResNet(RESNET_BLOCKS, num_classes=2, input_dimensions=RESNET_IMAGE_SIZE)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    cuda.empty_cache()
    model = model.to(DEVICE)
    
    total_batches = len(train)
    total_val_batches = len(test)
    for epoch in range(1, RESNET_EPOCHS):
        running_loss, acc = epoch_loop(train, model, optimizer, criterion, epoch, total_batches, show_progress=True)
        
        with no_grad():
            val_loss, val_acc = epoch_loop(test, model, optimizer, criterion, epoch, total_batches, show_progress=False)
            training_status(
                epoch,
                total_batches,
                running_loss / total_batches,
                acc / total_batches,
                total_batches,
                val_loss / total_val_batches,
                val_acc / total_val_batches
            )

# TODO: add tests
