import glob
import os

files_in = glob.glob('*.py')
files = []
for i in range(len(files_in)):
	if (files_in[i] != 'setup.py') and (files_in[i] != 'assemble.py'):
		files.append(files_in[i])
		
strop = 'cat '+' '.join([' ' + f for f in files])+' > ./prediction_functions/__init__.py'
os.system(strop)