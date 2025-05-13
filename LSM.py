###LSM (Least Squares Method)

import numpy as np
import matplotlib.pyplot as plt

x_years=np.array([1994,1995,1996,1997,1998,1999,2000,2001,2002,2003])
y_tones=np.array([67.052,68.008,69.083,72.024,73.400,72.063,74.669,74.487,74.065,76.777])

data_zip=list(zip(x_years,y_tones))
def linear_model(data):#y=kx+b
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
     
    sum_x=sum(data_x)
    sum_y=sum(data_y)
    sum_xy=sum([x*y for x,y in data])
    sum_x2=sum([x**2 for x in data_x])
    n=len(data_x)

    A=np.array([[n,sum_x],[sum_x,sum_x2]])
    b=np.array([sum_y,sum_xy])
    b,k=np.linalg.solve(A,b)
    return k,b    

k,b=linear_model(data_zip)
print("k=",k,"b=",b)

    