# Authors: Jeffrey Wang
# License: BSD 3 clause

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision.transforms as T
import torchvision.datasets as tvd
from torch.utils.data import DataLoader

class WideResNet:
	def __init__(self, input_shape=(3,32,32), n_classes=10,
					depth=16, width=8, lr=0.01, lr_decay=1,
					schedule=None, momentum=0.9, dropout=0.4, weight_decay=0.0005,
					epochs=100, batch_size=128, preprocess_method=None,
					seed=None):
		self.input_shape = input_shape
		self.n_classes = n_classes
		self.depth = depth
		self.width = width
		self.lr = lr
		self.lr_decay = lr_decay
		self.schedule = [] if schedule is None else list(map(int, schedule.split("|")))
		self.momentum = momentum
		self.dropout = dropout
		self.weight_decay = weight_decay
		self.epochs = epochs
		self.batch_size = batch_size
		self.preprocess_method = preprocess_method
		self.np_rng = np.random.RandomState(seed=seed)

	def fit(self, train, val=None):
		train = self.preprocess(train, train=True)
		if val is not None:
			val = self.preprocess(val)
		self.params_ = self._wideresnet_init()
		self.forward_ = self._wideresnet()

		def set_opt(lr):
			return SGD([v for v in self.params_.values() if v.requires_grad], lr,
						momentum=self.momentum, weight_decay=self.weight_decay)

		def calculate_correct(y_hat, y):
			_, p = torch.max(y_hat, -1)
			return (p == y).sum().item()

		lr = self.lr
		opt = set_opt(lr)
		for e in range(self.epochs):
			if e in self.schedule:
				lr *= self.lr_decay
				opt = set_opt(lr)
			avg_loss = 0
			avg_err = 0
			batches = tqdm(train, desc='Epoch %d/%d' %(e, self.epochs))
			for b, data in enumerate(batches):
				x, y = data
				if torch.cuda.is_available():
					x, y = x.cuda(), y.cuda()
				opt.zero_grad()
				y_hat = self.forward_(x, self.params_, train=True)
				loss = F.cross_entropy(y_hat, y)
				loss.backward()
				opt.step()
				err = 1 - calculate_correct(y_hat, y) / y.size(0)
				if b == 0:
					avg_loss = loss.item()
					avg_err = err
				avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
				avg_err = 0.9 * avg_err + 0.1 * err
				batches.set_postfix({'loss': '%.4f'%avg_loss, 'err': '%.4f'%avg_err})
			val_correct = 0
			val_total = 0
			val_loss = 0
			with torch.no_grad():
				for data in val:
					x, y = data
					if torch.cuda.is_available():
						x, y = x.cuda(), y.cuda()
					y_hat = self.forward_(x, self.params_, train=False)
					val_correct += calculate_correct(y_hat, y)
					val_total += y.size(0)
					val_loss += F.cross_entropy(y_hat, y).item() * y.size(0)
			val_err = 1 - val_correct / val_total
			val_loss /= val_total
			print("\tValidation - Loss: %.4f, Error: %.4f" % (val_loss, val_err))

	def preprocess(self, ds, method=None, train=False):
		transform = T.Compose([T.ToTensor()])
		if self.preprocess_method == "cifar10":
			mean = np.array([125.3, 123.0, 113.9]) / 255.0
			std = np.array([63.0, 62.1, 66.7]) / 255.0
			transform = T.Compose([transform, T.Normalize(mean, std)])
			if train:
				transform = T.Compose([
								T.Pad(4, padding_mode='reflect'),
								T.RandomHorizontalFlip(),
								T.RandomCrop(32), transform])
		ds = ds('.', train=train, download=True, transform=transform)
		return DataLoader(ds, self.batch_size, shuffle=train, num_workers=0,
					pin_memory=torch.cuda.is_available())

	def _wideresnet(self):
		assert ((self.depth - 4) % 6 == 0)
		n = (self.depth - 4) // 6

		def bn(x, p, base, train):
			return F.batch_norm(x, weight=p[base + '.weight'],
						bias=p[base + '.bias'],
						running_mean=p[base + '.running_mean'],
						running_var=p[base + '.running_var'],
						training=train)

		def unit(x, p, base, stride, train):
			y = F.relu(bn(x, p, base+'.bn0', train), inplace=True)
			y_a = F.conv2d(y, p[base+'.conv0.weight'], stride=stride, padding=1)
			y_a = F.relu(bn(y_a, p, base+'.bn1', train), inplace=True)
			if self.dropout > 0:
				y_a = F.dropout(y_a, p=self.dropout, training=train)
			y_a = F.conv2d(y_a, p[base+'.conv1.weight'], stride=1, padding=1)
			if base+'.convdim.weight' in p:
				return y_a + F.conv2d(y, p[base+'.convdim.weight'], stride=stride, padding=0)
			return y_a + x

		def stack(x, p, base, stride, train):
			for i in range(n):
				x = unit(x, p, base+'.unit%d'%i, stride if i == 0 else 1, train)
			return x

		def forward(x, p, train=False):
			y = F.conv2d(x, p['conv.weight'], stride=1, padding=1)
			y = stack(y, p, 'stack0', 1, train)
			y = stack(y, p, 'stack1', 2, train)
			y = stack(y, p, 'stack2', 2, train)
			y = F.relu(bn(y, p, 'bn', train))
			y = F.avg_pool2d(y, 8, 1, 0)
			y = y.view(y.size(0), -1)
			y = F.linear(y, p['dense.weight'], p['dense.bias'])
			return y

		return forward

	def _wideresnet_init(self):
		assert ((self.depth - 4) % 6 == 0)
		n = (self.depth - 4) // 6
		n_stages = [16, 16*self.width, 32*self.width, 64*self.width]
		def weight_init(t):
			return nn.init.kaiming_normal_(t)

		def conv_init(ni, no, k):
			return {'weight': weight_init(torch.empty(no, ni, k, k))}

		def bn_init(n):
			return {
					'weight': torch.rand(n),
					'bias': torch.zeros(n),
					'running_mean': torch.zeros(n),
					'running_var': torch.ones(n)
				}

		def linear_init(ni, no):
			return {'weight': weight_init(torch.empty(no, ni)), 'bias': torch.zeros(no)}

		def unit_init(ni, no):
			conv0 = {'conv0.'+k: v for k,v in conv_init(ni, no, 3).items()}
			conv1 = {'conv1.'+k: v for k,v in conv_init(no, no, 3).items()}
			bn0 = {'bn0.'+k: v for k,v in bn_init(ni).items()}
			bn1 = {'bn1.'+k: v for k,v in bn_init(no).items()}
			p = {**conv0, **conv1, **bn0, **bn1}
			if ni != no:
				p = {**p, **{'convdim.'+k: v for k,v in conv_init(ni, no, 1).items()}}
			return p

		def stack_init(ni, no, height):
			p = {}
			for i in range(height):
				pi = {'unit%d.' % i + k: v for k,v in unit_init(ni if i==0 else no, no).items()}
				p = {**p, **pi}
			return p

		wrn = {'conv.'+k: v for k,v in conv_init(self.input_shape[0], n_stages[0], 3).items()}
		wrn = {**wrn, **{'stack0.'+k: v for k,v in stack_init(n_stages[0], n_stages[1], n).items()}}
		wrn = {**wrn, **{'stack1.'+k: v for k,v in stack_init(n_stages[1], n_stages[2], n).items()}}
		wrn = {**wrn, **{'stack2.'+k: v for k,v in stack_init(n_stages[2], n_stages[3], n).items()}}
		wrn = {**wrn, **{'bn.'+k: v for k,v in bn_init(n_stages[3]).items()}}
		wrn = {**wrn, **{'dense.'+k: v for k,v in linear_init(n_stages[3], self.n_classes).items()}}
		wrn = {k: v.cuda() for k,v in wrn.items()}
		for k,v in wrn.items():
			if not k.endswith('running_mean') and not k.endswith('running_var'):
				v.requires_grad = True
		return wrn
