# Newton Divided Difference Interpolation

import numpy as np
import matplotlib.pyplot as pyp

############################## FUNCTIONS #############################

# Evaluate divided difference interpolant
def newtonEval(t,coefs,x):
    n = len(coefs)
    value = coefs[n-1]
    for i in range(n-2,-1,-1): # same as n-2, n-3, n-4, ..., 0
        value = value*(t-x[i]) + coefs[i]
    return value

# generate an array of points to plot a graph smoothly
def evaluate_graph(x, coefs):
    n = len(x)
    m = 10*n
    minx = min(x)
    maxx = max(x)
    t = np.arange(minx, maxx+1, (maxx-minx)/m)
    val = newtonEval(t, coefs, x)
    return (t, val)

# Set up divided difference coefficients
def newtonDD(x,y):
    n = len(x)

    # DD level 0
    # coefs[i] = y[i] for i=0,1,2,...,n-1
    coefs = [y[i] for i in range(n)] 
    
    # DD higher levels (bottom to top, overwrite lower entries as they are finished)
    for level in range(1,n): # 1,2,3,4, ... n-1
        for i in range(n-1,level-1,-1): #n-1, n-2, ..., level
            dx = x[i] - x[i-level] 
            if (dx==0): exit(2)
            coefs[i] = (coefs[i]-coefs[i-1])/dx
    return coefs

# print_poly helper
def sign_switch(num):
    num = str(num)
    if num[0] == '-':
        num = " - " + num[1:]
    else:
        num = " + " + num
    return num

# construct a string representing the polynomial
def print_poly(coefs, x_data, name, print_flag):
    P = f"{coefs[0]}"
    if print_flag:
        print(f"Interpolating Polynomial of {name}(x): \n" + "P(x) = " + P)
    for i in range(1, len(coefs)):
        addOn = sign_switch(coefs[i])
        for j in range(i):
            if x_data[j] == 0:
                addOn += " * (x - 0)"
            else:
                addOn += " * (x" + sign_switch(-x_data[j]) + ")"
        P += f"{addOn}"
        if print_flag:
            print(f"      {addOn}")
    if print_flag:
        print()
    return P

# evaluate NewtonDD polynomial
def eval_poly(P, x):
    P = eval(P)
    return P

# convert x-values into cos domain
def domain_cos(x_value):
    x_revised = []
    for x in x_value:
        s = 1
        x = x % (2*np.pi)
        if x > np.pi:
            x = 2*np.pi - x
        if x > np.pi/2:
            s = -1
            x = np.pi - x
        x_revised.append((x, s))
    return x_revised

# convert x-values into sin domain
def domain_sin(x_value):
    x_revised = []
    for x in x_value:
        sign = 1
        x = x % (2*np.pi)
        if x > np.pi:
            sign = -1
            x = 2*np.pi - x
        if x > np.pi/2:
            x = np.pi - x
        x_revised.append((x, sign))
    return x_revised

# generate chebysheve nodes
def chebyshev_nodes(a,b,n):
    default_nodes = [np.cos((2*i-1)*np.pi/(2*n)) for i in range(1, n+1)]
    nodes = [(b-a)/2*t+(b+a)/2 for t in default_nodes]
    return nodes

############################## PROBLEMS #############################

# generates the polynomials with specific interpolants
def interpolation(x_data, y_data, name, print_flag):
    # check length
    if len(x_data)==len(y_data):
        coefs = newtonDD(x_data, y_data)
    else:
        print("x and y are different sizes")
        exit()

    # store and display the polynomial
    P = print_poly(coefs, x_data, name, print_flag)
    return (P, coefs)

def get_cos_error(P, x_value, name, print_flag):
    # map x_value into cos1/2's domain
    x_revised = domain_cos(x_value)

    cos_true = [np.cos(x) for x in x_value]
    cos_esti = [sign*eval_poly(P, x) for (x, sign) in x_revised]
    cos_err = [abs(cos_true[i]-cos_esti[i]) for i in range(len(x_value))]

    if print_flag:
        # print and compare values and errors
        print("{:>10} {:>20} {:>20} {:>20}".\
            format("x", "cos(x)", name+"(x)", "Error"))
        for i in range(len(x_value)):
            print("{:10.1f} {:20.7f} {:20.7f} {:20.7f}".\
                format(x_value[i], cos_true[i], cos_esti[i], cos_err[i]))
        print()
    
    return cos_err

def compare_graph(graph1, graph2, nodes, name):
    # plot graphs
    pyp.figure(figsize=(8,4))
    pyp.plot(graph1[0], graph1[1], color="steelblue")
    pyp.plot(graph2[0], graph2[1], color="palevioletred")
    for i in range(len(nodes[0])):
        pyp.plot(nodes[0][i], nodes[1][i],'*', color = "navy")

    # make labels
    pyp.xlabel("x")
    pyp.ylabel("y")
    pyp.xlim(-np.pi, 3*np.pi)
    pyp.ylim(-1,1.1)
    pyp.title(name+"(x) Interpolation vs cos(x)")
    pyp.legend(["cos(x)", name+"(x)"], loc="upper right")
    pyp.show()

def compare_error(interval, err1, err2, name1, name2):
    # plot error points
    pyp.figure(figsize=(10,4))
    pyp.plot(interval, err1, color="indianred")
    pyp.plot(interval, err2, color="cadetblue")

    # make labels
    pyp.xlabel("x")
    pyp.ylabel("Errors")
    pyp.xlim(-np.pi, 3*np.pi)
    pyp.ylim(0, 0.025)
    pyp.title(f"Error Comparison of {name1}(x) and {name2}(x)")
    pyp.legend([f"{name1} errors(x)", f"{name2} errors(x)"], loc="upper right")
    pyp.show()

def analyze_cos_P(x_value, interpolants, name, print_flag):
    # generate the interpolated polynomial
    cos_x_data = interpolants
    cos_y_data = [np.cos(x) for x in cos_x_data] 
    cos_P, coefs = interpolation(cos_x_data, cos_y_data, name, print_flag)

    # generate and print x, cos, cos_P, and error
    get_cos_error(cos_P, x_value, name, print_flag)

    # plot graph
    if print_flag:
        # pre-plot values for cos
        cosx = np.arange(-np.pi, 3*np.pi, 0.1)
        cosy = np.cos(cosx)

        # pre-plot values for cos_P
        t, val = evaluate_graph(cos_x_data, coefs)

        # generate the graph of cos and cos_P
        compare_graph((cosx, cosy), (t, val), (cos_x_data, cos_y_data), name)

    return cos_P

############################## MAIN #############################

def main():
    # points to be evaluated using polynomials
    x_value = [1, 2, 3, 4, 14, 1000]
    x_value_neg = [-x for x in x_value]
    x_value_neg.reverse()
    x_value = x_value_neg + x_value

    # equally spaced interpolants
    interpolants = [0, np.pi/4, np.pi/2]
    cos1 = analyze_cos_P(x_value, interpolants, "cos1", True)

    # chebyshev interpolants
    interpolants = chebyshev_nodes(0, np.pi/2, 3)
    cos2 = analyze_cos_P(x_value, interpolants, "cos2", True)

    # compare errors in cos1 and cos2
    interval = np.linspace(-np.pi, 3*np.pi, num=500)
    err1 = get_cos_error(cos1, interval, "cos1", False)
    err2 = get_cos_error(cos2, interval, "cos2", False)
    compare_error(interval, err1, err2, "cos1", "cos2")

if __name__ == "__main__":
    main()
