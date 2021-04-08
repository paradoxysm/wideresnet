import os
from setuptools import setup, find_packages

def read(*paths):
    """
	Build a file path from *paths* and return the contents.
	"""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
	name='wideresnet',
	version='1.0.0',
	description='Wide Residual Networks in Keras and PyTorch',
	long_description=(read('README.md') + '\n\n'),
	long_description_content_type="text/markdown",
	url='http://github.com/paradoxysm/wideresnet',
	author='paradoxysm',
	author_email='paradoxysm.dev@gmail.com',
	license='BSD-3-Clause',
	packages=find_packages(),
	install_requires=[
		torch==1.8.1+cu102,
		torchvision==0.9.1+cu102,
		tensorflow=2.4.1,
		tqdm,
		numpy,
    ],
	python_requires='>=3.5, <4',
	classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
	'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8',
    ],
	keywords=['python', 'ml', 'neural-networks', 'keras', 'tensorflow', 'pytorch', 'reproducibility'],
    zip_safe=True)
