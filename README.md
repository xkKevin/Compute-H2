## The nonlinear regression coefficient h2 is calculated as follows: 
1. The average of the signals is set to zero. Xi and Yi are the amplitude values of the ith samples of the X and Y signal. 
2. A scattergram is made: the amplitude values of Yare plotted as a function of the corresponding amplitude values of X. 
3. The ordinate is split into equal-sized bins. Within each bin, the average of the y-values is calculated and called Qj for bin j. The midpoint of bin j is called Pj. 
4. The points (Pj, Qj) with (Pj+1, Qj+1) is called gj(x) and the whole piecewise curve is called f(x). The deviation of each sample from the curve is calculated: Yi - f(Xi) for (Xi, Yi). The sum of the square of the deviations is called the unexplained variance. h2 is the variance in all the y-values (called the total variance) minus the unexplained variance, all divided by the total variance.

### You can also view my codes from this website: https://nbviewer.jupyter.org/ 