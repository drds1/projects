import datetime
import time
import os
pwd = os.getcwd()+'/'


print('the time is...')
time = datetime.datetime.now()
print(time)


#append to file
with open(pwd+"time.txt", "a") as myfile:
    myfile.write(np.str(time)+'\n')


