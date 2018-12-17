#import required dependencies locally

import json, os, sys, logging
import pandas as pd, numpy as np
import sqlalchemy
from tqdm import tqdm
from pandas.io.sql import SQLTable
import matplotlib.pyplot as plt

from utils import sql_utils, math_utils

pd.set_option("display.max_rows", 200)