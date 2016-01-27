#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import matplotlib.pyplot as plt

fname = 'log_inbin_reconfig.txt'

train_x = []
train_loss0 = []
train_acc0 = []
test_x = []
test_acc0 = []

with open(fname, 'r') as f:
    for line in f.readlines():
        epoch = int(re.search('epoch:([\-0-9]+?);', line).groups()[0])
        if 'train' in line and 'loss' in line:
            train_x.append(epoch)
            loss = float(re.search('train mean loss0=([0-9\.\-]+?);', line).groups()[0])
            train_loss0.append(loss)
        if 'train' in line and 'accuracy' in line:
            acc = float(re.search('train mean accuracy0=([0-9\.\-]+?);', line).groups()[0])
            train_acc0.append(acc)
        if 'test' in line:
            test_x.append(epoch)
            acc = float(re.search('test mean accuracy0=([0-9\.\-]+?);', line).groups()[0])
            if acc < 0:
                acc = test_acc0[0]
            test_acc0.append(acc)

train_x.insert(0, 0)
print(train_x)
print(train_loss0)
train_loss0.insert(0, None)
print(train_acc0)
train_acc0.insert(0, None)

print(test_x)
print(test_acc0)

fig, ax1 = plt.subplots()

ax1.set_xlabel('epoch')
ax1.set_xlim((-1, len(train_x)))
ax1.set_ylabel('loss')
ax1.plot(train_x, train_loss0, label='train loss')

ax2 = ax1.twinx()

ax2.plot(train_x, train_acc0, c='g', label='train accuracy')
ax2.plot(test_x, test_acc0, c='r', label='test accuracy')
ax2.set_xlim((-1, len(test_x)))
ax2.set_ylabel('accuracy')

ax1.legend(loc=0)
ax2.legend(loc=9)

plt.show()