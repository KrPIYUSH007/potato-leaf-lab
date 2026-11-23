"""Frozen transfer backbones must keep their weights and running statistics fixed."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from leaflab.models import TransferClassifier


@pytest.mark.parametrize('architecture', ['regnet_y_400mf', 'mobilenet_v2'])
def test_head_training_preserves_frozen_backbone(architecture):
    torch.set_num_threads(2)
    model = TransferClassifier(architecture, pretrained=False).train()
    frozen = next(p for p in model.parameters() if not p.requires_grad)
    frozen_before = frozen.detach().clone()
    normalization = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    mean_before = normalization.running_mean.clone()
    head_before = model.head.bias.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 7)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    optimizer.step()
    assert not torch.equal(head_before, model.head.bias)
    assert torch.equal(frozen_before, frozen)
    assert torch.equal(mean_before, normalization.running_mean)
    assert frozen.grad is None

    head_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.set_stage('fine_tune')
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > head_count
    assert not normalization.training
    model.eval()
    assert not any(m.training for m in model.modules())
