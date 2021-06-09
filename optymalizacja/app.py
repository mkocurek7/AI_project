import numpy as np
import minhelper 

#f(x,y)= (1-x)^2 + 100(y - x^2)^2

def rosenbrock(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.

def main():
    target = [1, 1]
    easy_init = [2, 2]
    hard_init = [-1.2, 1]
    minhelper.show_minimization_results(
            rosenbrock, target, easy_init, hard_init)

if __name__ == '__main__':
    main()