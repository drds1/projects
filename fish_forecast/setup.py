from setuptools import setup

setup(name='fish_forecast',
      version='0.1.1',
      description='Ok-ish forecasting code',
      url='https://github.com/dstarkey23/projects/tree/master/fish_forecast',
      author='dstarkey23',
      author_email='ds207@st-andrews.ac.uk',
      license='MIT',
      packages=['fish_forecast'],
      install_requires=[
      'scikit-learn',
      'scipy',
      'prediction_functions',
      ],
      zip_safe=False)