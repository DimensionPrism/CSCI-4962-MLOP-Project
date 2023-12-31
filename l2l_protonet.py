#!/usr/bin/env python3

import pdb
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels


def pairwise_distances_logits(a, b, metric='euclidean', eps=1e-8):
    n = a.shape[0]
    m = b.shape[0]
    a_expanded = a.unsqueeze(1).expand(n, m, -1)
    b_expanded = b.unsqueeze(0).expand(n, m, -1)
    
    if metric == "euclidean":
        logits = -((a_expanded - b_expanded)**2).sum(dim=2)
    if metric == "mahalanobis":
        diff = a_expanded - b_expanded
        covariance_matrix = torch.cov(b.T)
        covariance_matrix += torch.eye(covariance_matrix.size(0)).cuda()
        inv_covariance_matrix = torch.linalg.inv(covariance_matrix)
        logits = -torch.einsum('ijk,kl,ijl->ij', diff, inv_covariance_matrix, diff)
    if metric == 'itakura_saito':
        a_expanded = a_expanded.add(eps)
        b_expanded = b_expanded.add(eps)
        
        ratio = torch.clamp(a_expanded / b_expanded, min=0.1, max=10)
        
        itakura_saito_dist = ratio - torch.log(ratio) - 1
        logits = -itakura_saito_dist.sum(dim=2)
        
        max_logit = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - max_logit
    if metric == 'generalized_i_divergence':
        a_expanded = a_expanded.add(eps)
        b_expanded = b_expanded.add(eps)
        
        log_ratio = torch.log(a_expanded / b_expanded)
        divergence = a_expanded * log_ratio - (a_expanded - b_expanded)
        logits = -divergence.sum(dim=2)
    if metric == "kl_divergence":
        a_expanded = a_expanded.add(eps)
        b_expanded = b_expanded.add(eps)
        
        # Compute the KL divergence
        # KL(P || Q) = sum(P(x) * log(P(x)/Q(x)))
        kl_div = a_expanded * torch.log2(a_expanded / b_expanded)
        logits = -kl_div.sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.CNN4Backbone(
            hidden_size=hid_dim,
            channels=x_dim,
            max_pool=True,
       )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class ViTEncoder(nn.Module):
    def __init__(self, image_size=84, patch_size=4, embedding_dim=1024, num_heads=4, num_layers=1, hidden_dim=1024, flatten_channels=True):
        super(ViTEncoder, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = int((image_size / patch_size)**2)
        self.embedding_dim = embedding_dim
        self.flatten_channels = flatten_channels

        self.linear_projection = nn.Linear(self.patch_size * self.patch_size * 3, self.embedding_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(8)
        self.classifier = nn.Linear(self.embedding_dim * 8, self.embedding_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if self.flatten_channels:
            x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        x = self.linear_projection(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)

        x = x.flatten(1)
        x = self.classifier(x)
        return x

def fast_adapt(model, batch, ways, shot, query_num, metric='euclidean', device=None):
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()
    
    logits = pairwise_distances_logits(query, support, metric=metric)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=250)
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=30)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--metric', type=str, default='euclidean')
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    if args.model == 'cnn':
        model = Convnet()
    elif args.model == 'transformer':
        model = ViTEncoder()
    model.to(device)

    path_data = './data'
    train_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='train', download=True)
    valid_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='validation', download=True)
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root=path_data, mode='test', download=True)

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.Taskset(train_dataset, task_transforms=train_transforms)
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),
        KShots(valid_dataset, args.test_query + args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.Taskset(
        valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=200,
    )
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        NWays(test_dataset, args.test_way),
        KShots(test_dataset, args.test_query + args.test_shot),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.Taskset(
        test_dataset,
        task_transforms=test_transforms,
        num_tasks=2000,
    )
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
            batch = next(iter(train_loader))

            loss, acc = fast_adapt(model,
                                   batch,
                                   args.train_way,
                                   args.shot,
                                   args.train_query,
                                   metric=args.metric,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print('epoch {}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(valid_loader):
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=args.metric,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(
            epoch, n_loss/loss_ctr, n_acc/loss_ctr))

    loss_ctr = 0
    n_acc = 0

    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(model,
                               batch,
                               args.test_way,
                               args.test_shot,
                               args.test_query,
                               metric=args.metric,
                               device=device)
        loss_ctr += 1
        n_acc += acc
    print('Test Accuracy: {:.2f}'.format(n_acc/loss_ctr * 100))
        