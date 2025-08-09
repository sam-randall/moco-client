import os

import pytorch_lightning as pl
import torch
from torchvision.datasets import MNIST
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from src.early_exit_model import EarlyExitModel
from tqdm import tqdm 
from typing import Tuple
import time
class LitMNIST(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # 28x28 -> 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), # 26x26 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2),         # 24x24 -> 12x12
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

def train_model(train_set, save_path: str = "mnist_model_train.ckpt", fast_dev: bool = False):

    model = LitMNIST(lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        log_every_n_steps=10,
        fast_dev_run = fast_dev
    
    )

    loader = DataLoader(train_set, batch_size = 64, num_workers=6)
    trainer.fit(model, loader, )

    trainer.save_checkpoint(save_path)

    return model


class EmbeddingGenerator:
    def __init__(self, model: nn.Sequential, layer: int):
        self.model = model
        self.layer = layer
    
    def generate(self, data: torch.Tensor, batch_size = 1000):
        embeddings = []
        predictions = []
        for i in tqdm(range(0, len(data), batch_size)):
            embedding = self.model[:self.layer](data[i: i + batch_size].type(torch.FloatTensor))
            pred = self.model(data[i: i + batch_size].type(torch.FloatTensor))
            embeddings.append(embedding)
            predictions.append(pred)
        embeddings = torch.vstack(embeddings)
        array = embeddings.detach().cpu().numpy()
        preds = torch.vstack(predictions)
        logits = preds.detach().cpu().numpy()
        return array, logits


def get_embeddings_from_sequential_model_at_layer(sequential_model: nn.Sequential,
                                data: torch.Tensor,
                                layer: int,
                                batch_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Computes embeddings from sequential model at layer `layer`.

    Args:
        sequential_model (nn.Sequential): Model to extract embeddings from.
        data (torch.Tensor): Dataset to embed
        layer (int): layer l in Model to extract from.
        batch_size (int): Batch Size running data through the model

    Returns:
        out (Tuple[embeddings, predictions]): tuple containing embeddings and predictions
    
    '''
    generator = EmbeddingGenerator(sequential_model, layer)
    embedding, logits = generator.generate(data, batch_size)
    return embedding, logits

def main():

    transform = transforms.ToTensor()
    train = MNIST(root=".", train=True, download=True, transform=transform)
    test = MNIST(root = ".", train = False, download = True, transform=transform)
  
    # Train Model
    if os.path.exists("mnist_model_train.ckpt"):
        model = LitMNIST.load_from_checkpoint("mnist_model_train.ckpt")
    else:
        model = train_model(train, fast_dev=False)
    train_data = train.data[:, torch.newaxis, :, :]

    # Make predictions on train dataset.
    with torch.no_grad():
        predictions = []
        for i in tqdm(range(0, len(train), 1000)):
            p = model(train_data[i: i + 1000].type(torch.FloatTensor)).cpu()
            predictions.append(p)

        predictions = torch.vstack(predictions).argmax(axis = 1)

    predictions = predictions.detach().cpu().numpy()

    ## Now to use MOCO, we need to extract embeddings from the neural network
    ## and analyze those embeddings so as to fashion short-circuit rules.
    ## We've done some analysis work ahead of time to choose from which layer
    ## to extract embeddings from. We welcome you to experiment or contact us.
    
    # Extract Embeddings using helper methods.
    # This really can happen under the hood.
    # Wrap model in EarlyExitModel.

    # Call API 
    sequential_model = model.model
    
    sequential_model

    eem = EarlyExitModel(sequential_model)
    print("Generating Embeddings now...")
    embedding, p = get_embeddings_from_sequential_model_at_layer(sequential_model, train_data, 2)
    print("Computing short circuit rules.")
    N = embedding.shape[0]
    embedding = embedding.reshape((N, -1))
    if os.path.exists('mnist_rules.json'):
        import json
        with open('mnist_rules.json') as f:
            d = json.load(f)
            d = json.loads(d)
        result = eem.apply_rules_from_json_string(d)
        print(result)

        start = time.time()
        for i in range(0, len(train), 1000):
            with torch.no_grad():
                eem.predict(train_data[i: i + 1000])
        end = time.time()
        print(end - start)

        start =	time.time()
        for i in range(0, len(train), 1000):
            with torch.no_grad():
                sequential_model(train_data[i: i + 1000])
        end = time.time()
        print(end - start)
    else:
        email_address = "sam.randall5@gmail.com"
        result = eem.compute_short_circuit_rules(embedding, predictions, 1e-7, email_address)

        print(result)
if __name__ == "__main__":
    main()

