#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import matplotlib.pyplot as plt

fname = 'log_train_inhand.txt'

train_x = []
train_loss = []
train_acc = []
test_x = []
test_acc = []

with open(fname, 'r') as f:
    for line in f.readlines():
        epoch = int(re.search('epoch:([\-0-9]+?);', line).groups()[0])
        print(epoch)
        if 'train' in line and 'loss' in line:
            train_x.append(epoch)
            loss = float(re.search('train mean loss=(.+?);', line).groups()[0])
            train_loss.append(loss)
        elif 'train' in line and 'accuracy' in line:
            acc = float(re.search('train mean accuracy=(.+?);', line).groups()[0])
            train_acc.append(acc)
        elif 'test' in line and 'accuracy' in line:
            test_x.append(epoch)
            acc = float(re.search('test mean accuracy=(.+?);', line).groups()[0])
            if acc < 0:
                acc = test_y[0]
            test_acc.append(acc)

print(train_x)
print(train_loss)
print(train_acc)

print(test_x)
print(test_acc)

fig, ax1 = plt.subplots()

ax1.set_xlabel('epoch')
ax1.set_xlim((-1, len(train_x)))
ax1.set_ylabel('loss')
ax1.plot(train_x, train_loss, label='train loss')

ax2 = ax1.twinx()
ax2.plot(train_x, train_acc, c='g', label='train accuracy')
ax2.plot(test_x, test_acc, c='r', label='test accuracy')
ax2.set_xlim((-1, len(test_x)))
ax2.set_ylabel('accuracy')

# ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
# ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
ax1.legend(loc=0)
ax2.legend(loc=9)

# plt.show()
plt.savefig('log_train_inhand.jpg')
