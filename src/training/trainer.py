from dataclasses import dataclass
from typing import Optional

from src.model.model import Model
from src.model.loss import Loss
from src.model.optimizer import Optimizer
from src.data.dataloader import Dataloader
#
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         logits = model.forward(batch)
#         loss = criterion.forward(logits, targets)
#         delta_Z = criterion.backward()
#         model.backward(delta_Z)
#         optimizer.step(model.params(), model.grads())


@dataclass(slots=True, init=False)
class Trainer:
    model: Model
    loss: Loss
    optimizer: Optimizer
    train_dataloader: Dataloader
    val_dataloader: Optional[Dataloader]
    test_dataloader: Optional[Dataloader]
