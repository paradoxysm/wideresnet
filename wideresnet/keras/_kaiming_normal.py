# Authors: Jeffrey Wang
# License: BSD 3 clause

from tensorflow.keras.initializers import VarianceScaling

class KaimingNormal(VarianceScaling):
	def __init__(self, seed=None):
		super(KaimingNormal, self).__init__(
				scale=2., mode='fan_in', distribution='untruncated_normal', seed=seed)

	def get_config(self):
		return {'seed': self.seed}
