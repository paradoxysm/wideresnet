# Modified from tensorflow/python/keras/optimizer_v2/gradient_descent.py 
# Original Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License")

# Authors of Modified Work: Jeffrey Wang
# License: BSD 3 clause

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, state_ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

class SGDtorch(optimizer_v2.OptimizerV2):
	_HAS_AGGREGATE_GRAD = True

	def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False,
					weight_decay=0.0, name="SGDtorch", **kwargs):
		super(SGDtorch, self).__init__(name, **kwargs)
		self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
		self._set_hyper("decay", self._initial_decay)

		self._momentum = False
		if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
			self._momentum = True
		if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
			raise ValueError("`momentum` must be between [0, 1].")
		self._set_hyper("momentum", momentum)

		self._weight_decay = False
		if isinstance(weight_decay, ops.Tensor) or callable(weight_decay) or weight_decay > 0:
			self._weight_decay = True
		if isinstance(weight_decay, (int, float)) and (weight_decay < 0 or weight_decay > 1):
			raise ValueError("`weight_decay` must be between [0, 1].")
		self._set_hyper("weight_decay", weight_decay)

		self.nesterov = nesterov

	def _create_slots(self, var_list):
		slots = []
		if self._momentum : slots.append("momentum")
		if self._weight_decay : slots.append("weight_decay")
		if len(slots) > 0:
			for var in var_list:
				for s in slots:
					self.add_slot(var, s)

	def _prepare_local(self, var_device, var_dtype, apply_state):
		super(SGDtorch, self)._prepare_local(var_device, var_dtype, apply_state)
		extra = ["momentum", "weight_decay"]
		for v in extra:
			apply_state[(var_device, var_dtype)][v] = array_ops.identity(
				self._get_hyper(v, var_dtype))

	def _resource_apply_dense(self, grad, var, apply_state=None):
		var_device, var_dtype = var.device, var.dtype.base_dtype
		coefficients = ((apply_state or {}).get((var_device, var_dtype))
							or self._fallback_apply_state(var_device, var_dtype))

		if self._momentum:
			v = self.get_slot(var, "momentum")
			g = self.get_slot(var, "weight_decay")
			lr_t = coefficients["lr_t"]
			m = coefficients["momentum"]
			w_d = coefficients["weight_decay"]
			if self._weight_decay:
				g_t = state_ops.assign(g, grad + w_d * var, use_locking=self._use_locking)
			else: g_t = state_ops.assign(g, grad, use_locking=self._use_locking)
			v_t = state_ops.assign(v, m * v + g_t, use_locking=self._use_locking)
			if self.nesterov : g_n = state_ops.assign_add(g_t, m * v_t, use_locking=self._use_locking)
			else : g_n = state_ops.assign(g_t, v_t, use_locking=self._use_locking)
			var_update = state_ops.assign_sub(var, lr_t * g_n, use_locking=self._use_locking)
			return tf.group(*[g_t, v_t, g_n, var_update])
		else:
			return gen_training_ops.ResourceApplyGradientDescent(
				var=var.handle,
				alpha=coefficients["lr_t"],
				delta=grad,
				use_locking=self._use_locking)

	def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
		raise NotImplementedError("Resource Apply Sparse")
		if self._momentum:
			return super(SGD, self)._resource_apply_sparse_duplicate_indices(
				grad, var, indices, **kwargs)
		else:
			var_device, var_dtype = var.device, var.dtype.base_dtype
			coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
							or self._fallback_apply_state(var_device, var_dtype))

		return gen_resource_variable_ops.ResourceScatterAdd(
			resource=var.handle,
			indices=indices,
			updates=-grad * coefficients["lr_t"])

	def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
		# This method is only needed for momentum optimization.
		raise NotImplementedError("Resource Apply Sparse")
		var_device, var_dtype = var.device, var.dtype.base_dtype
		coefficients = ((apply_state or {}).get((var_device, var_dtype))
						or self._fallback_apply_state(var_device, var_dtype))

		momentum_var = self.get_slot(var, "momentum")
		return gen_training_ops.ResourceSparseApplyKerasMomentum(
			var=var.handle,
			accum=momentum_var.handle,
			lr=coefficients["lr_t"],
			grad=grad,
			indices=indices,
			momentum=coefficients["momentum"],
			use_locking=self._use_locking,
			use_nesterov=self.nesterov)

	def get_config(self):
		config = super(SGD, self).get_config()
		config.update({
			"learning_rate": self._serialize_hyperparameter("learning_rate"),
			"decay": self._serialize_hyperparameter("decay"),
			"momentum": self._serialize_hyperparameter("momentum"),
			"weight_decay": self._serialize_hyperparameter("weight_decay"),
			"nesterov": self.nesterov,
		})
		return config
