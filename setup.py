from setuptools import setup, find_packages

setup(
  name = 'byol-pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.3.1',
  license='MIT',
  description = 'Self-supervised contrastive learning made simple',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/byol-pytorch',
  keywords = ['self-supervised learning', 'artificial intelligence'],
  install_requires=[
      'torch>=1.6',
      'kornia>=0.4.0'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)