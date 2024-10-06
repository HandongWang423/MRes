import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.optimize import minimize

def add_val_label(val):
    return "$%.2f^{+%.2f}_{-%.2f}$"%(val[0],abs(val[1]),abs(val[2]))

def find_crossings(graph, yval, spline_type="cubic", spline_points=1000, remin=True, return_all_intervals=False):

    # Build spline
    f = interp1d(graph[0],graph[1],kind=spline_type)
    x_spline = np.linspace(graph[0].min(),graph[0].max(),spline_points)
    y_spline = f(x_spline)
    spline = (x_spline,y_spline)

    # Remin
    if remin:
        x,y = graph[0],graph[1]
        if y_spline.min() <= 0:
            y = y-y_spline.min()
            y_spline -= y_spline.min()
            # Add new point to graph
            x = np.append(x, x_spline[np.argmin(y_spline)])
            y = np.append(y, 0.)
            # Re-sort
            i_sort = np.argsort(x)
            x = x[i_sort]
            y = y[i_sort]
            graph = (x,y)

    # Extract bestfit
    bestfit = graph[0][graph[1]==0]

    crossings, intervals = [], []
    current = None

    for i in range(len(graph[0])-1):
        if (graph[1][i]-yval)*(graph[1][i+1]-yval) < 0.:
            # Find crossing as inverse of spline between two x points
            mask = (spline[0]>graph[0][i])&(spline[0]<=graph[0][i+1])
            f_inv = interp1d(spline[1][mask],spline[0][mask])

            # Find crossing point for catch when yval is out of domain of spline points (unlikely)
            if yval > spline[1][mask].max(): cross = f_inv(spline[1][mask].max())
            elif yval <= spline[1][mask].min(): cross = f_inv(spline[1][mask].min())
            else: cross = f_inv(yval)

            # Add information for crossings
            if ((graph[1][i]-yval) > 0.)&( current is None ):
                current = {
                    'lo':cross,
                    'hi':graph[0][-1],
                    'valid_lo': True,
                    'valid_hi': False
                }
            if ((graph[1][i]-yval) < 0.)&( current is None ):
                current = {
                    'lo':graph[0][0],
                    'hi':cross,
                    'valid_lo': False,
                    'valid_hi': True
                }
            if ((graph[1][i]-yval) < 0.)&( current is not None ):
                current['hi'] = cross
                current['valid_hi'] = True
                intervals.append(current)
                current = None

            crossings.append(cross)

    if current is not None:
        intervals.append(current)

    if len(intervals) == 0:
        current = {
            'lo':graph[0][0],
            'hi':graph[0][-1],
            'valid_lo': False,
            'valid_hi': False
        }
        intervals.append(current)

    for interval in intervals:
        interval['contains_bf'] = False
        if (interval['lo']<=bestfit)&(interval['hi']>=bestfit): interval['contains_bf'] = True

    for interval in intervals:
        if interval['contains_bf']:
            val = (bestfit, interval['hi']-bestfit, interval['lo']-bestfit)

    if return_all_intervals:
        return val, intervals
    else:
        return val

## Neural network tools
def get_batches(arrays, batch_size=None, randomise=False, include_remainder=True):
    length = len(arrays[0])
    idx = np.arange(length)

    if randomise:
        np.random.shuffle(idx)

    n_full_batches = length // batch_size
    is_remainder = (length % batch_size > 0)

    if is_remainder and include_remainder:
        n_batches = n_full_batches + 1
    else:
        n_batches = n_full_batches

    for i_batch in range(n_batches):
        if i_batch < n_full_batches:
            batch_idx = idx[i_batch*batch_size:(i_batch+1)*batch_size]
        else:
            batch_idx = idx[i_batch*batch_size:]

        arrays_batch = [torch.Tensor(array[batch_idx]) for array in arrays]
        yield arrays_batch
def get_total_loss(model, loss, X, y, mean_over_batch=False):
    eval_batch_size = min(1024*2**8,len(X))
    losses = [] # contain loss from every batch

    with torch.no_grad():
        for X_tensor, y_tensor in get_batches([X, y], eval_batch_size):
            output = model(X_tensor)
            losses.append(loss(output, y_tensor).item()*len(output))

        if mean_over_batch:
            mean_loss = sum(losses) / len(losses)
        else:
            mean_loss = sum(losses) / len(X)

    return mean_loss

def get_total_lossW(model, loss, X, y,w, mean_over_batch=False):
    eval_batch_size = min(1024*2**8,len(X))
    losses = [] # contain loss from every batch

    with torch.no_grad():
        for X_tensor, y_tensor, w_tensor in get_batches([X, y,w], eval_batch_size):
            output = model(X_tensor)
            losses.append(loss(output, y_tensor, w_tensor).item()*len(output))

        if mean_over_batch:
            mean_loss = sum(losses) / len(losses)
        else:
            mean_loss = sum(losses) / len(X)

    return mean_loss

def get_total_lossM(model, loss, X, y,w, mean_over_batch=False):
    eval_batch_size = min(1024*2**8,len(X))
    
    losses = [] # contain loss from every batch

    with torch.no_grad():
        for X_tensor, y_tensor, w_tensor in get_batches([X, y,w], eval_batch_size):
            losses.append(loss(X_tensor, y_tensor, w_tensor,model).item()*len(X_tensor))

        if mean_over_batch:
            mean_loss = sum(losses) / len(losses)
        else:
            mean_loss = sum(losses) / len(X)

    return mean_loss




def get_network_output(df, features, model):
    with torch.no_grad():
        X = df[features].to_numpy() #Now check what corrections are for test_df

        eval_batch_size = min(1024*2**8,len(df))
        outputs = []

        for X_tensor, in get_batches([X], eval_batch_size):
              outputs.append(model(X_tensor).numpy())

        output = np.concatenate(outputs)

    return output

def NLL(s, b, theta_init, rho_s, rho_b):
    rho_s, rho_b = torch.tensor(rho_s), torch.tensor(rho_b)
    counts_exp = s*rho_s + b*rho_b
    asimov = theta_init['s']*rho_s + theta_init['b']*rho_b
    return -1*(asimov*torch.log(counts_exp)-counts_exp).sum()