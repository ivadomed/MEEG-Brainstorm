#!/usr/bin/env python

"""
This script is used to do warmups steps on the learning rate during training.
Inspired by: `"Attention Is All You Need"
<https://arxiv.org/pdf/1606.08415v3.pdf>`_.

Usage: type "from learning_rate_warmup import <class>" to use class.

Contributors: Ambroise Odonnat.
"""


class NoamOpt():

    """
    Warmup method inspired by:
    `<http://nlp.seas.harvard.edu/2018/04/03/attention.html>`_.
    """

    def __init__(self, max_lr, warmup, optimizer):

        """
        Args:
            max_lr (float): Maximum value of learning rate after warmup steps.
            warmup (int): Warmup steps value.
            optimizer (Optimizer): Adaptative Optimizer chosen for training.
        """

        self.max_lr = max_lr
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0

    def step(self):

        """
        Update parameters and rate.
        """

        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self):

        """
        Compute rate.
        """

        return min(self.max_lr, self._step * self.warmup ** (-1.5))
