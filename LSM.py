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
def parabolia_modek(data):
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    sum_x=sum(data_x)
    sum_y=sum(data_y)
    sum_xy=sum([x*y for x,y in data])
    sum_x2=sum([x**2 for x in data_x])
    sum_x3=sum([x**3 for x in data_x])
    sum_x4=sum([x**4 for x in data_x])
    sum_x2y=sum([x**2*y for x,y in data])
    n=len(data_x)
    A=np.array([[n,sum_x,sum_x2],[sum_x,sum_x2,sum_x3],[sum_x2,sum_x3,sum_x4]])
    b=np.array([sum_y,sum_xy,sum_x2y])
    b,a,c=np.linalg.solve(A,b)
    return a,b,c
def cubic_model(data):
    data_x = [data[i][0] for i in range(len(data))]
    data_y = [data[i][1] for i in range(len(data))]
    sum_x=sum(data_x)
    sum_x2=sum([x**2 for x in data_x])
    sum_x3=sum([x**3 for x in data_x])
    sum_x4=sum([x**4 for x in data_x])
    sum_x5=sum([x**5 for x in data_x])
    sum_x6=sum([x**6 for x in data_x])
    sum_xy=sum([x*y for x,y in data])
    sum_x2y=sum([x**2*y for x,y in data])
    sum_x3y=sum([x**3*y for x,y in data])
    sum_y=sum(data_y)
    n=len(data_x)
    A=np.array([[n,sum_x,sum_x2,sum_x3],[sum_x,sum_x2,sum_x3,sum_x4],[sum_x2,sum_x3,sum_x4,sum_x5],[sum_x3,sum_x4,sum_x5,sum_x6]])
    b=np.array([sum_y,sum_xy,sum_x2y,sum_x3y])
    b,a,c,d=np.linalg.solve(A,b)
    return a,b,c,d
    
def kb_print():   
    k_linear,b_linear=linear_model(data_zip)
    print("k=",k_linear,"b=",b_linear)

    k_parabolia,b_parabolia,c_parabolia=parabolia_modek(data_zip)
    print("k=",k_parabolia,"b=",b_parabolia,"c=",c_parabolia)

    k_cubic,b_cubic,c_cubic,d_cubic=cubic_model(data_zip)
    print("k=",k_cubic,"b=",b_cubic,"c=",c_cubic,"d=",d_cubic)
    
if __name__=="__main__":
    kb_print()