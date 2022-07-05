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
    Implementation inspired by:
    `<http://nlp.seas.harvard.edu/2018/04/03/attention.html>`_.
    """

    def __init__(self,
                 optimizer,
                 warmup=2e3):

        """
        Args:
            optimizer (Optimizer): Adaptative Optimizer chosen for training.
            warmup (int): Warmup steps value.
        """

        self.optimizer = optimizer
        self.warmup = warmup
        self._step = 0
        self._rate = 0

    def step(self):

        """
        Update parameters and rate.
        """

        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):

        """
        Compute rate.
        """

        if step is None:
            step = self._step

        return min(1e-3, step * self.warmup ** (-1.5))
