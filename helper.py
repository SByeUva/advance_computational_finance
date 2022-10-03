import numpy as np
from scipy import odr

def GBM_Euler(T, S, sigma, r, M):
    '''
    Function that simulates a stock price under general brownian motion
    T = days in future
    S = initial stock price
    sigma = volatility
    r = drift
    M = amount of movements
    '''
    S_all = []
    S_all.append(S)
    dt = T/M
    
    Zm = np.random.normal(size=M)
    for i in range(M-1):
        S_all.append(S_all[i] + r * S_all[i] * dt + sigma * S_all[i] * np.sqrt(dt) * Zm[i]) 
    return S_all


def value_option_schwarz(M,K,path_matrix, r, realizations, option="call", poly_choice="laguerre"):
    '''
    Longstaff-Scharwz option pricer
    '''
    
    stopping_rule = np.zeros(path_matrix.shape)
    # save payoffs for later use
    if option == "call":
        exercise_value = np.maximum(path_matrix-K,0)
        stopping_rule[:,-1] = np.where(path_matrix[:,-1]-K>0, 1, 0)
    else:
        exercise_value = np.maximum(K-path_matrix,0)
        stopping_rule[:,-1] = np.where(K-path_matrix[:,-1]>0, 1, 0)
        
    exercise_value[0,:] = 0

    for time in range(1,M-1):
        # get X at time step and Y at time step+1 (Regress now) 
        if option == "call":
            X = np.where(path_matrix[:,M-time-1]>K, path_matrix[:,M-time-1], 0)
            Y = np.where(path_matrix[:,M-time-1]>K, path_matrix[:,M-time], 0)
        else:
            X = np.where(path_matrix[:,M-time-1]<K, path_matrix[:,M-time-1], 0)
            Y = np.where(path_matrix[:,M-time-1]<K, path_matrix[:,M-time], 0)
            
        X_nonzero = X[X>0]
        Y_nonzero = Y[X>0]
        if option=="call":
            Y_nonzero = np.maximum(Y_nonzero-K,0)*np.exp(-r)
        else:
            Y_nonzero = np.maximum(K-Y_nonzero,0)*np.exp(-r)
        
        
        if len(Y_nonzero!=0):
            try:
                poly = np.polynomial.laguerre.Laguerre.fit(X_nonzero, Y_nonzero, 2)
            except:
                print(X_nonzero)
                print(Y_nonzero)
            final_y = np.zeros(len(X_nonzero))
            for i, val in enumerate(X_nonzero): 
                final_y[i] = poly(val)

            ## Compare excerise with continuation
            ex_cont = np.zeros((len(X_nonzero), 2))

            if option == "call":
                ex_cont[:,0] = X_nonzero - K
            else:
                ex_cont[:,0] = K - X_nonzero
            ex_cont[:,1] = final_y
                
            j=0
            for i in range(len(X)):
                if X[i] > 0:
                    if ex_cont[j,0] > ex_cont[j,1]:
                        stopping_rule[i,:] = 0
                        stopping_rule[i,M-time-1] = 1
                    j+=1
        else:
            print(f"time: {time}")
            print("No path found")
    return stopping_rule * exercise_value

def poly(x):
    return (-1.813 * x**2 + 2.983 *x -1.07)

def poly2(x):
    return (2.038 -3.335 *x + 1.356 * x**2)

def value_option_schwarz_test(M,K,path_matrix, r, realizations, option="call"):
    option_cash_flow_matrix = np.zeros(path_matrix.shape)
    if option == "call":
        option_cash_flow_matrix[:,-1] = np.maximum(path_matrix[:,M-1]-K, 0)
    else:
        option_cash_flow_matrix[:,-1] = np.maximum(K-path_matrix[:,M-1], 0)
    for time in range(M-2):
        if option == "call":
            X = np.where(path_matrix[:,M-time-2]>K, path_matrix[:,M-time-2], 0)
            Y = np.where(path_matrix[:,M-time-2]>K, option_cash_flow_matrix[:,M-time-1], 0) *np.exp(-r)
        else:
            X = np.where(path_matrix[:,M-time-2]<K, path_matrix[:,M-time-2], 0)
            Y = np.where(path_matrix[:,M-time-2]<K, option_cash_flow_matrix[:,M-time-1], 0) *np.exp(-r)

        X_nonzero = X[X>0]
        Y_nonzero = Y[X>0]
        
        if time == 0:
            final_y = poly(X_nonzero)
        if time == 1:
            final_y = poly2(X_nonzero)
        ## Compare excerise with continuation
        ex_cont = np.zeros((len(X_nonzero), 2))
        
        if option == "call":
            ex_cont[:,0] = X_nonzero - K
        else:
            ex_cont[:,0] = K - X_nonzero
        ex_cont[:,1] = final_y
        
        cash_flow = np.zeros((realizations, 2))
        
        j=0
        for i in range(len(X)):
            if X[i] > 0:
                if ex_cont[j,0] > ex_cont[j,1]:
                    cash_flow[i, 0] = ex_cont[j,0] 
                else:
                    cash_flow[i,1] = ex_cont[j,1]

                j+=1
        
        # save answer of time
        for i, ans in enumerate(cash_flow[:,0]):
            if ans!=0:
                option_cash_flow_matrix[i,:] = 0
                option_cash_flow_matrix[i,M-time-2] = ans
    return option_cash_flow_matrix


def value_option_bermudan(M,K,path_matrix, r, realizations, exercise_dates, option="call", poly_choice="laguerre"):
    '''
    Longstaff-Scharwz option pricer
    '''
    
    option_cash_flow_matrix = np.zeros(path_matrix.shape)
    if option == "call":
        option_cash_flow_matrix[:,-1] = np.maximum(path_matrix[:,M-1]-K, 0)
    else:
        option_cash_flow_matrix[:,-1] = np.maximum(K-path_matrix[:,M-1], 0)
    
    
    for time in range(M-2):
        
        if time in exercise_dates:
        
            if option == "call":
                X = np.where(path_matrix[:,M-time-2]>K, path_matrix[:,M-time-2], 0)
                Y = np.where(path_matrix[:,M-time-2]>K, option_cash_flow_matrix[:,M-time-1], 0) *np.exp(-r)
            else:
                X = np.where(path_matrix[:,M-time-2]<K, path_matrix[:,M-time-2], 0)
                Y = np.where(path_matrix[:,M-time-2]<K, option_cash_flow_matrix[:,M-time-1], 0) *np.exp(-r)

            X_nonzero = X[X>0]
            Y_nonzero = Y[X>0]

            if poly_choice == "laguerre":
                poly = np.polynomial.laguerre.Laguerre.fit(X_nonzero, Y_nonzero, 3)
                final_y = np.zeros(len(X_nonzero))
                for i, val in enumerate(X_nonzero): 
                    final_y[i] = poly(val)
            else:
                poly = odr.polynomial(2)
                data = odr.Data(X_nonzero ,Y_nonzero)
                model = odr.ODR(data, poly)
                output = model.run()
                final = np.poly1d(output.beta[::-1])
                final_y = final(X_nonzero)
            ## Compare excerise with continuation
            ex_cont = np.zeros((len(X_nonzero), 2))

            if option == "call":
                ex_cont[:,0] = X_nonzero - K
            else:
                ex_cont[:,0] = K - X_nonzero
            ex_cont[:,1] = final_y

            cash_flow = np.zeros((realizations, 2))

            j=0
            for i in range(len(X)):
                if X[i] > 0:
                    if ex_cont[j,0] > ex_cont[j,1]:
                        cash_flow[i, 0] = ex_cont[j,0] 
                    else:
                        cash_flow[i,1] = ex_cont[j,1]

                    j+=1

            # save answer of time
            for i, ans in enumerate(cash_flow[:,0]):
                if ans!=0:
                    option_cash_flow_matrix[i,:] = 0
                    option_cash_flow_matrix[i,M-time-2] = ans
    
    
    return option_cash_flow_matrix