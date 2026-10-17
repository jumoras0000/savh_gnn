"""Régression : un batch ignoré ne doit pas appeler GradScaler sans gradients."""
import pytest

torch = pytest.importorskip("torch")


def test_skipped_final_batch_does_not_step_empty_optimizer():
    from src.training.train_phase3 import train_one_epoch

    class Batch:
        num_graphs = 1

        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            return {"efficacy": self.weight * torch.tensor([[batch.value]])}

    class Criterion:
        def __call__(self, predictions, targets):
            value = predictions["efficacy"]
            return value.mean(), {}

    class CheckingScaler:
        def __init__(self):
            self.steps = 0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            assert any(p.grad is not None for group in optimizer.param_groups
                       for p in group["params"])
            self.steps += 1
            optimizer.step()

        def update(self):
            pass

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = CheckingScaler()
    batches = [(Batch(value), {"efficacy": torch.tensor([[1.0]])})
               for value in (1.0, 1.0, float("nan"))]

    train_one_epoch(model, batches, optimizer, Criterion(), torch.device("cpu"),
                    grad_clip=1.0, epoch=1, scaler=scaler, accum_steps=2)

    assert scaler.steps == 1
