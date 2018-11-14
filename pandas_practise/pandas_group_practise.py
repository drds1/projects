import pandas as pd
import numpy as np




people = pd.DataFrame(np.random.randn(5, 5), columns=['a', 'b', 'c', 'd', 'e'], 
index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])



#group mapping 
mapping = np.where(people['b'] > people['a'], 'Group1', 'Group2')


mapping = 
#apply the grouping. Get group by object
people_group = people.groupby(mapping)