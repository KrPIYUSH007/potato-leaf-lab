"""Seven-category transfer models with explicit head and fine-tuning stages."""
import argparse
import json
from pathlib import Path

import torch
from torchvision.models import (MobileNet_V2_Weights, RegNet_Y_400MF_Weights,
                                mobilenet_v2, regnet_y_400mf)

from .download import CLASSES

WEIGHT_PRESETS = {
    'regnet_y_400mf': RegNet_Y_400MF_Weights.IMAGENET1K_V2,
    'mobilenet_v2': MobileNet_V2_Weights.IMAGENET1K_V2,
}
BUILDERS = {'regnet_y_400mf': regnet_y_400mf, 'mobilenet_v2': mobilenet_v2}


class TransferClassifier(torch.nn.Module):
    def __init__(self, architecture, pretrained=True, cache_dir='runs/torch-hub'):
        super().__init__()
        if architecture not in BUILDERS:
            raise ValueError(f'Unknown transfer architecture: {architecture}')
        self.architecture = architecture
        self.weights = WEIGHT_PRESETS[architecture]
        if pretrained:
            torch.hub.set_dir(str(Path(cache_dir)))
        self.network = BUILDERS[architecture](weights=self.weights if pretrained else None)
        if architecture == 'regnet_y_400mf':
            self.network.fc = torch.nn.Linear(self.network.fc.in_features, len(CLASSES))
        else:
            self.network.classifier[-1] = torch.nn.Linear(
                self.network.classifier[-1].in_features, len(CLASSES))
        self.set_stage('head')

    @property
    def head(self):
        if self.architecture == 'regnet_y_400mf':
            return self.network.fc
        return self.network.classifier[-1]

    def fine_tune_blocks(self):
        if self.architecture == 'regnet_y_400mf':
            return [self.network.trunk_output[-1]]
        return list(self.network.features[-4:])

    def set_stage(self, stage):
        if stage not in ('head', 'fine_tune'):
            raise ValueError('Stage must be head or fine_tune')
        self.stage = stage
        self.network.requires_grad_(False)
        self.head.requires_grad_(True)
        if stage == 'fine_tune':
            for block in self.fine_tune_blocks():
                block.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen batch-normalization statistics fixed when the trainer calls train().
        self.network.eval()
        self.head.train(mode)
        if self.stage == 'fine_tune':
            for block in self.fine_tune_blocks():
                block.train(mode)
        if self.architecture == 'mobilenet_v2':
            self.network.classifier.train(mode)
        return self

    def forward(self, images):
        return self.network(images)

    def validation_transform(self):
        return self.weights.transforms()


def main():
    parser = argparse.ArgumentParser(description='Verify pretrained seven-category transfer models')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', default='reports/transfer_models.json')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    results = {}
    for architecture in BUILDERS:
        model = TransferClassifier(architecture).eval().to(args.device)
        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 224, 224, device=args.device))
        if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
            raise ValueError(f'Invalid model outputs for {architecture}')
        head_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.set_stage('fine_tune')
        results[architecture] = {
            'status': 'architecture_verified_training_pending',
            'weights': str(model.weights), 'classes': list(CLASSES),
            'input_shape': [1, 3, 224, 224], 'output_shape': list(logits.shape),
            'device': args.device, 'total_parameters': sum(p.numel() for p in model.parameters()),
            'head_stage_trainable_parameters': head_parameters,
            'fine_tune_stage_trainable_parameters': sum(
                p.numel() for p in model.parameters() if p.requires_grad),
            'validation_transform': str(model.validation_transform()),
        }
        del model
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
