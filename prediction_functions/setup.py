from setuptools import setup

setup(name='prediction_functions',
      version='0.1.0',
      description='reuqirements for fish_forecasting code',
      url='https://github.com/dstarkey23/projects/tree/master/prediction_functions',
      author='dstarkey23',
      author_email='ds207@st-andrews.ac.uk',
      license='MIT',
      packages=['prediction_functions'],
      install_requires=[
      'scikit-learn',
      'scipy',
      ],
      long_description=open('README.txt').read(),
      zip_safe=False)