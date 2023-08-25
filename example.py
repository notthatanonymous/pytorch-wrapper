import torch
import torchvision
import math
import random
import numpy as np
import os

from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from pytorch_wrapper import modules, System
from pytorch_wrapper import evaluators as evaluators
from pytorch_wrapper.loss_wrappers import GenericPointWiseLossWrapper
from pytorch_wrapper.training_callbacks import EarlyStoppingCriterionCallback, NumberOfEpochsStoppingCriterionCallback


class MNISTDatasetWrapper(Dataset):
    def __init__(self, is_train):
        self.dataset = MNIST(
            'data/mnist/',
            train=is_train,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

    def __getitem__(self, index):
        return {'input': self.dataset[index][0], 'target': self.dataset[index][1]}

    def __len__(self):
        return len(self.dataset)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out_mlp = modules.MLP(
            input_size=980,
            num_hidden_layers=1,
            hidden_layer_size=128,
            hidden_activation=nn.ReLU,
            output_size=10,
            output_activation=None
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        return self.out_mlp(x)




train_val_dataset = MNISTDatasetWrapper(is_train=True)
test_dataset = MNISTDatasetWrapper(is_train=False)

# Use 10% of the training dataset as validation.
val_size = math.floor(0.1 * len(train_val_dataset))
train_val_indexes = list(range(len(train_val_dataset)))
random.seed(12345)
random.shuffle(train_val_indexes)
train_indexes = train_val_indexes[val_size:]
val_indexes = train_val_indexes[:val_size]

train_dataloader = DataLoader(
    train_val_dataset,
    sampler=SubsetRandomSampler(train_indexes),
    batch_size=128
)

val_dataloader = DataLoader(
    train_val_dataset,
    sampler=SubsetRandomSampler(val_indexes),
    batch_size=128
)

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)



model = Model()

last_activation = nn.Softmax(dim=-1)
if torch.cuda.is_available():
    system = System(model, last_activation=last_activation, device=torch.device('cuda'))
else:
    system = System(model, last_activation=last_activation, device=torch.device('cpu'))


loss_wrapper = GenericPointWiseLossWrapper(nn.CrossEntropyLoss())
evals = {

    # 'prec': evaluators.MultiClassPrecisionEvaluator(average='macro'),
    # 'rec': evaluators.MultiClassRecallEvaluator(average='macro'),
    # 'f1': evaluators.MultiClassF1Evaluator(average='macro'),
    'acc': evaluators.MultiClassAccuracyEvaluator()

}

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, system.model.parameters()))

os.makedirs('tmp', exist_ok=True)
_ = system.train(
    loss_wrapper,
    optimizer,
    train_data_loader=train_dataloader,
    evaluators=evals,
    evaluation_data_loaders={
        'val': val_dataloader
    },
    callbacks=[NumberOfEpochsStoppingCriterionCallback(3)
        # EarlyStoppingCriterionCallback(
        #     patience=3,
        #     evaluation_data_loader_key='val',
        #     evaluator_key='f1',
        #     tmp_best_state_filepath='tmp/mnist_tmp_best.weights'
        # )
    ]
)


results = system.evaluate(test_dataloader, evals)


print(results)

print(type(results))

# for r in results:
#     print(results[r])

score = str(results['acc']).split(': ')[1].split('%')[0]
print(f"\n\n\nScore: {score}\n\n\n")
