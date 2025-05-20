from glob import glob

import torch
import torchvision
from torch import nn
from transformers import BertModel
from transformers import BertForSequenceClassification

from .isomaxplus import IsoMaxPlusLossFirstPart
from .datasets import IMAGE_DATASETS, TABULAR_DATASET, TEXT_DATASETS


class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model, hparams=None):
        super().__init__()
        if hparams is None:
            hparams = {'last_layer_dropout': .0}
        self.model = model
        self.n_outputs = model.config.hidden_size
        classifier_dropout = (
            hparams['last_layer_dropout'] if hparams['last_layer_dropout'] != 0. else model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, 'pooler_output'):
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])


def get_backbone(dataset_name, pretrained_in=False, model_name='resnet50'):
    if dataset_name in IMAGE_DATASETS:
        weights = 'IMAGENET1K_V2' if pretrained_in else None
        match model_name:
            case 'resnet50':
                model = torchvision.models.resnet50(weights=weights)
            case 'resnet152':
                model = torchvision.models.resnet152(weights=weights)
            case 'resnext101':
                model = torchvision.models.resnext101_64x4d(weights=weights)
        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        emb_dim = model.fc.in_features
    elif dataset_name in TEXT_DATASETS:
        backbone = BertFeatureWrapper(BertModel.from_pretrained('bert-base-uncased'))
        emb_dim = backbone.n_outputs
    else:
        raise ValueError(f'Dataset {dataset_name} not supported.')
    return backbone, emb_dim


def get_model(dataset_name, num_classes, train_mode='full',
              pretrained_path=None, pretrained_in=False, loss_name='ce',
              model=None, verbose=True, resume=False,
              model_name='resnet50'):
    if resume:
        assert pretrained_path is not None

    ckpt = None
    if pretrained_path is not None:
        if '*' in pretrained_path:
            pretrained_path = glob(pretrained_path)
            assert len(pretrained_path) == 1
            pretrained_path = pretrained_path[0]
        ckpt = torch.load(pretrained_path, map_location="cpu")

    if model is None:  # Stage 0 or training resume
        backbone, emb_dim = get_backbone(dataset_name, pretrained_in, model_name=model_name)
        if (((ckpt is not None) and ('prototypes' in [k.split('.')[-1] for k in ckpt.keys()])) or
                (loss_name == 'isomax')):
            head = IsoMaxPlusLossFirstPart(emb_dim, num_classes)
        else:
            head = nn.Linear(emb_dim, num_classes)
        if model is None:
            model = nn.Sequential(backbone, nn.Flatten(), head)
    else:  # ongoing training with model is passed from the previous stage
        backbone = torch.nn.Sequential(*list(model.children())[:-1])  # extract the current backbone
        head = None

    if ckpt is not None:
        try:
            model.load_state_dict(ckpt, strict=True)
        ## weight mismatched
        except Exception as e:
            print(e)
            print('Load partial state dict...')
            model.load_state_dict(ckpt, strict=False)

    if head is None:
        if not resume:  # Create a new a head for training the next stage
            if hasattr(model[-1], 'prototypes'):
                emb_dim = model[-1].prototypes.shape[1]
            else:
                emb_dim = model[-1].in_features
        print(f'Building a new classifier ({loss_name})...')
        if loss_name == 'isomax':
            head = IsoMaxPlusLossFirstPart(emb_dim, num_classes)  # head is reset (even when loading checkpoint)
        elif loss_name == 'ce':
            head = nn.Linear(emb_dim, num_classes, bias=False)
        else:
            raise ValueError('loss_name must be either "ce" or "isomax"')
        model = nn.Sequential(backbone, nn.Flatten(), head)

    if train_mode == 'freeze':
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()  # Freeze everything including batchnorm

    if verbose:
        pytorch_total_params = sum(p.numel() for p in model.parameters()) / 1e6
        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {pytorch_total_trainable_params}/{pytorch_total_params:.2f}M')

    return model
