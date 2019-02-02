from setuptools import setup

setup(name='prediction_functions',
      version='1.1.2',
      description='prediction function library used in fish fc',
      #url='https://github.com/dstarkey23/projects/tree/master/fish_forecast',
      #author='dstarkey23',
      #author_email='ds207@st-andrews.ac.uk',
      license='MIT',
      #packages=['fish_forecast'],
      install_requires=[
      'scikit-learn',
      'scipy',
      #'prediction_functions',
      ],
      zip_safe=False)