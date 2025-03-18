""" grad_based.py (c) Yang Yang, 2024-2025
This file implements an SGD-based optimization algorithm to train LSTM.
"""

import torch
from matplotlib import pyplot as plt
from torch import nn
from blocks.lstm import LSTM
from _global import info
from typing import List, Dict
from demo import save_model

show_gradient_explosion = False


def demo(optimizer, num_epochs: int, model: LSTM, train_x, train_y,
         test_x, test_y, plot, method_name, save=False) -> Dict[str, List[float] or str]:
    criterion = nn.MSELoss()

    train_loss, val_loss = [criterion(model(train_x), train_y).item()], [criterion(model(test_x), test_y).item()]
    info(f'{method_name["name"]}: At the beginning: train_loss: {train_loss[0]}; val_loss: {val_loss[0]}')

    gradients = {name: [] for name, _ in model.named_parameters()}

    for epoch in range(num_epochs):
        model.train()

        outputs = model(train_x)
        loss = criterion(outputs, train_y)

        optimizer.zero_grad()
        loss.backward()

        if show_gradient_explosion:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name].append(param.grad.norm().cpu().numpy())

        optimizer.step()

        train_loss_t = loss.item()

        info(f"{method_name['name']}: Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss_t}")
        train_loss.append(train_loss_t)

        model.eval()
        with torch.no_grad():
            predictions = model(test_x)
            test_loss_t = criterion(predictions, test_y).item()
            val_loss.append(test_loss_t)
            info(f"{method_name['name']}: Test Loss: {test_loss_t}")

    if plot:
        plt.plot(list(range(num_epochs + 1)), val_loss if show_gradient_explosion else train_loss)
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if show_gradient_explosion:
            plt.yscale('log')
        plt.show()

    if save:
        save_model(method_name['name'], model)

    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'gradients': gradients
    }


def sgd_demo(num_epochs: int, model: LSTM, train_x, train_y,
             test_x, test_y, lr, plot=False, save=False) \
        -> Dict[str, List[float] or str]:
    if show_gradient_explosion:
        lr = 7.4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    method_name = {'name': 'SGD'}
    results = demo(optimizer, num_epochs, model, train_x, train_y, test_x, test_y, plot, method_name, save)
    results.update(method_name)
    return results


def adam_demo(num_epochs: int, model: LSTM, train_x, train_y,
              test_x, test_y, lr, plot=False, save=False) \
        -> Dict[str, List[float] or str]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    method_name = {'name': 'Adam'}
    results = demo(optimizer, num_epochs, model, train_x, train_y, test_x, test_y, plot, method_name, save)
    results.update(method_name)
    return results


def adagrad_demo(num_epochs: int, model: LSTM, train_x, train_y,
                 test_x, test_y, lr, plot=False, save=False) \
        -> Dict[str, List[float] or str]:
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    method_name = {'name': 'Adagrad'}
    results = demo(optimizer, num_epochs, model, train_x, train_y, test_x, test_y, plot, method_name, save)
    results.update(method_name)
    return results
