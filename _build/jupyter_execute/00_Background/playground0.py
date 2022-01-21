# Play Ground

import numpy as np
import matplotlib.pyplot as plt
import math

Plot a sinusoid and exponential:

from math import pi

radians_to_degrees = 180.0/pi

x1vals = np.arange(0, 6*pi, 0.2)
x2vals = np.cos(x1vals)
x3vals = np.exp(-x1vals/(2*pi))

if False:
    print(x1vals)
    print(x2vals)
    print(x3vals)

plt.scatter(x1vals, x2vals)
plt.scatter(x1vals, x3vals,color="red")

# You can also set the title and axis labels this way
plt.title("Exponential curve and friend")
plt.ylabel("y-value")
plt.xlabel("x-value")

Try to make sure you always label your plots. And if we give you a plot with unlabeled axes or no title, tell us! We need to fix it.


Plot a noisy exponential and a noisy sinusoid:

def noisifyAbs(yvals_in):
    gamma = 0.5
    noise = gamma*np.max(yvals_in)*(np.random.random(yvals_in.shape)-0.5)
    yvals_out = yvals_in + noise
    return(yvals_out)

def noisifyRel(yvals_in):
    gamma = 0.5
    noise = gamma*np.max(yvals_in)*(np.random.random(yvals_in.shape)-0.5)
    yvals_out = []
    for ii in range(0, len(noise)):
        yvals_out.append(yvals_in[ii]*(1+noise[ii]))
    return(yvals_out)

radians_to_degrees = 180.0/math.pi

x1vals = np.arange(0, 6*math.pi, 0.5)
x2_clean = np.cos(x1vals)
x3_clean = np.exp(-x1vals/(2*math.pi))

x2_noisyAbs = noisifyAbs( x2_clean )
x3_noisyAbs = noisifyAbs( x3_clean )
x2_noisyRel = noisifyRel( x2_clean )
x3_noisyRel = noisifyRel( x3_clean )

if False:
    print(x2_noisyAbs)
    print(x3_noisyAbs)
    print(x2_noisyRel)
    print(x3_noisyRel) 

fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot(   x1vals,    x2_clean, color="black",linestyle="solid",linewidth=1)
axes.plot(   x1vals,    x3_clean, color="black",linestyle="solid",linewidth=1)
plt.scatter( x1vals, x2_noisyAbs, color="red"     )
plt.scatter( x1vals, x3_noisyAbs, color="green"   )
plt.scatter( x1vals, x2_noisyRel, color="blue"    )
plt.scatter( x1vals, x3_noisyRel, color="magenta" )

plt.title("Fancy scatter plot with colors")
plt.ylabel("y value")
plt.xlabel("x value")

Take a closer look at the properties of the noise.  Which noise model is more reasonable?

fig = plt.figure()
axes = fig.add_subplot(111)

## Change this to True to see x2 (sinusoid)
## Change this to False to see x3 (exponential)
show_x2 = True

if show_x2:
    axes.plot(x1vals, x2_clean, color="black", linestyle="solid", linewidth=1)   
    axes.scatter(x1vals, x2_noisyAbs, color="red")
    axes.scatter(x1vals, x2_noisyRel, color="blue")
else:
    axes.plot(x1vals, x3_clean, color="black", linestyle="solid", linewidth=1)
    axes.scatter(x1vals, x3_noisyAbs, color="green")
    axes.scatter(x1vals, x3_noisyRel, color="magenta")
    
plt.title("slightly less fancy scatter plot")
plt.ylabel("y value")
plt.xlabel("x value")