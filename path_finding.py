import numpy as np
import functools
import pandas as pd
import os
import copy
import operator




def find_neighbors(x, limit_x=[1, 548], limit_y=[1, 421]):
    xx = [(x[0]+i_i, x[1]+i_j) for i_i in [-1, 0, 1] for i_j in [-1, 0, 1] if abs(i_i)+abs(i_j)==1]
    
    xx = list(filter(lambda i: i[0]>=limit_x[0] and i[0]<=limit_x[1] and i[1]>=limit_y[0] and i[1]<=limit_y[1], xx))
    return xx



def pointwise_compare(x):
    x2 = np.concatenate([[np.nan], x[:-1]])
    return x!=x2

def _reconcile(s1, s2):
    pts = np.unique(np.sort(np.concatenate([s1._x, s2._x])))
    # Handle case when endpoints are inf
    cpts = pts.copy()
    cpts[0] = min(np.min(cpts[1:]), 0.) - 1
    cpts[-1] = max(np.max(cpts[:-1]), 0.) + 1
    mps = (cpts[1:] + cpts[:-1]) / 2.
    return [(pts, s(mps)) for s in (s1, s2)]


def _same_support(s1, s2):
    return np.all(s1._x[[0, -1]] == s2._x[[0, -1]])


def require_compatible(f):
    @functools.wraps(f)
    def wrapper(self, other, *args, **kwargs):
        if isinstance(other, StepFunction) and not _same_support(self, other):
            raise TypeError("Step functions have different support: %s vs. %s" % (
                self._x[[0, -1]], other._x[[0, -1]]))
        return f(self, other, *args, **kwargs)
    return wrapper


class StepFunction:
    '''A step function.'''

    def __init__(self, x, y):
        '''Initialize step function with breakpoints x and function values y.
        x and y are arrays such that
            f(z) = y[k], x[k] <= z < x[k + 1], 0 <= k < K.
        Thus, len(x) == len(y) + 1 and domain of f is (x[0], x[K]).
        '''
        if len(x) != 1 + len(y):
            raise RuntimeError("len(x) != 1 + len(y)")
        self._x = np.array(x)
        self._y = np.array(y, dtype='float')
        self._compress()

    @property
    def K(self):
        '''The number of steps.'''
        return len(self._y)

    def _compress(self):
        # Combine steps which have equal values
     #   ny = np.concatenate([[np.nan], self._y, [np.nan]])
     #   ys = np.diff(ny) != 0
        ys = np.concatenate([pointwise_compare(self._y), [True]])
        self._x = self._x[ys]
        self._y = self._y[ys[:-1]]

    def _binary_op(self, other, op, desc):
        if isinstance(other, StepFunction):
            (s1_x, s1_y), (s2_x, s2_y) = _reconcile(self, other)
            return StepFunction(s1_x, op(s1_y, s2_y))
        # Fall back to normal semantics otherwise
        return StepFunction(self._x, op(self._y, other))

    def __add__(self, other):
        return self._binary_op(other, operator.add, "add")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary_op(other, operator.sub, "subtract")

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return self._binary_op(other, operator.mul, "multiply")

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self._binary_op(other, operator.div, "divide")

    def __rdiv__(self, other):
        return (self ** -1) * other

    # Unary operations

    def __neg__(self):
        return StepFunction(self._x, -self._y)

    def __pow__(self, p):
        return StepFunction(self._x, pow(self._y, p))

    def __abs__(self):
        return StepFunction(self._x, abs(self._y))

    # Equality and comparison operators

    @require_compatible
    def __eq__(self, other):
        return (np.array_equal(self._x, other._x) and 
                np.array_equal(self._y, other._y))

    @require_compatible
    def __lt__(self, other):
        diff = other - self
        return np.all(diff._y > 0)

    @require_compatible
    def __le__(self, other):
        diff = other - self
        return np.all(diff._y >= 0)

    @require_compatible
    def __gt__(self, other):
        return -self < -other

    @require_compatible
    def __ge__(self, other):
        return -self <= -other

    def __call__(self, s):
        return self._y[np.searchsorted(self._x, s, side="right") - 1]

    def __repr__(self):
        return "StepFunction(x=%s, y=%s)" % (repr(self._x), repr(self._y))

    def integral(self):
        nz = self._y != 0
        d = np.diff(self._x)
        return (d[nz] * self._y[nz]).sum()
    
    def find_interval_value(self, x_interval):
        # [a, b)
     
        y = np.arange(np.argwhere(self._x<=x_interval[0])[-1], np.argwhere(self._x<x_interval[1])[-1]+1)
        yy = self._y[y]
        yyy = np.sort(np.unique(np.append(self._x[y], x_interval)))
        yyy = yyy[np.all(np.array([yyy>=x_interval[0], yyy<=x_interval[1]]), axis=0)]
        return np.array([yyy, yy])
   
    def extend_domain(self, new_value):
 
        if new_value < self._x[0]:
            self._x = np.insert(self._x, 0, new_value)
            self._y = np.insert(self._y, 0, float("Inf"))
            self.__init__(self._x, self._y)
        elif new_value > self._x[0]:
            self._x = np.append(self._x, new_value)
            self._y = np.append(self._y, float("Inf"))
            #self.__init__(self._x, self._y)
        else:
            pass



def hvdistance(x, y):
    return np.sum(np.abs(np.array(x) - np.array(y)))




class vertex_i:
    def __init__(self, vi, vs, ve, td, ta):
        self.vi = vi
        self.Ti = np.array([hvdistance(vi, vs)*2, ta])
        self.Si = np.array([ta, ta])
        self.TAUi = None
        self.negihbor_Sij = {}
        self.ta = ta
        self.td = td
        self.ve = ve
    #def update_Si(self):
        
    def update_TAUi_Si(self, interval_x):
        f = StepFunction(self.gi_d, self.gi_v)
        y = f.find_interval_value(interval_x)
        self.TAUi = y[1].min()
        self.TAUi2 = y[1].min() + hvdistance(self.vi, self.ve)
        # self.Si = np.array([y[0][np.argmin(y[1])], self.ta])

        self.Si = np.array([min(self.gi_d[np.append(self.gi_v == self.TAUi, [False])]), self.ta])
    
    
    def create_negihbor(self, ta):
        
        x = find_neighbors(self.vi, limit_x=[1, 548], limit_y=[1, 421])
        y = [[ta-2, ta-2]]*len(x)
        
        self.negihbor_Sij = dict(zip(x, y))
    
    def create_gi(self):
        self.gi_d = np.array([self.td, self.ta])
        self.gi_v = np.array([float('Inf')])
    
    def update_gi(self, i_x, i_y):
          
        i_x = np.array(i_x)
        i_y = np.array(i_y)
       
        f1 = StepFunction(self.gi_d, self.gi_v)
        
        f2 = StepFunction(i_x, i_y)
        
        if 0 not in i_x:
            f2.extend_domain(self.td)
        if 1080 not in i_x:
            f2.extend_domain(self.ta)
        
        x1 = np.sort(np.unique(np.append(self.gi_d, i_x)))
        y1 = [min(f1(i), f2(i)) if i in i_x else f1(i) for i in x1[:-1]]
        
        f = StepFunction(x1, y1)
        
        self.gi_d = f._x
        self.gi_v = f._y
    

def compute_minimum_cost(v_s, v_e, t_d, t_a, df):

    Q = {}
    QQ = []
    Q[v_s] = vertex_i(v_s, v_s, v_e, t_d, t_a)
    Q[v_s].gi_v = np.array([0])
    Q[v_s].gi_d = np.array([t_d, t_a])
    t = Q[v_s].Ti
    Q[v_s].update_TAUi_Si([t_d, t_a])
    Q[v_s].create_negihbor(t_a)
    
    
    v_i = v_s
    #run_times = 1
    while v_i != v_e:
        
        for i in find_neighbors(v_i, limit_x=[1, 548], limit_y=[1, 421]):
            # if (Q[v_i].Si[0] < (t_a - 2)) and (Q[v_i].TAUi != float('Inf')):
            if (Q[v_i].Si[0] < (t_a - 2)):
                R_ij = [Q[v_i].Si[0], t_a-2]
                 
               
                x = [R_ij[0], Q[v_i].negihbor_Sij[i][0]]
                
                
                domain_ij = [x[0]+2, x[1]+2]
        
            
              
                if domain_ij[0] < domain_ij[1]:
                    f_ij = df[(df['xid'] == i[0]) & (df['yid'] == i[1])]
                    f_ij.sort_values(by='hour', ascending=True, inplace=True)
                
              
                    f_ij = StepFunction(np.append(f_ij.hour, t_a), np.where(f_ij.wind>=15, 1440, 0))
                    
                    
                    f_ij = f_ij.find_interval_value(domain_ij)
                
                    update_d = f_ij[0]
                    update_v = f_ij[1] + Q[v_i].TAUi
                
                    Q[v_i].negihbor_Sij[i] = R_ij
                
                    if i not in [iii for iii in Q]:
                        Q[i] = vertex_i(i, v_s, v_e, t_d, t_a)
                        Q[i].create_gi()
                        Q[i].create_negihbor(t_a)
                    
                    domain_j = [Q[i].Ti[0], Q[i].Si[0]]
                    
                
                    
                    if Q[i].Ti[0] < Q[i].Si[0]:
                        Q[i].update_gi(update_d, update_v)

                
                        #run_times += 1
                        #print run_times
                        #print 'update %s in %s runs' % (i, run_times)
                
                        Q[i].update_TAUi_Si(domain_j)
                
 
                    if i not in [iii for iii, jjj in QQ]:
                        QQ.append((i, Q[i].TAUi2))
                              
                
                
        if (Q[v_i].Ti[0] < Q[v_i].Si[0]):
            
            domain_i = [Q[v_i].Ti[0], Q[v_i].Si[0]]
            
            Q[v_i].update_TAUi_Si(domain_i)
            #print domain_i
            #print Q[v_i].gi_d
            #print Q[v_i].gi_v
            #print '----------'
            
            QQ.append((v_i, Q[v_i].TAUi2))
            
        
        QQ = sorted(QQ, key=operator.itemgetter(1), reverse=False)
        
        v_i = QQ.pop(0)[0]
    return Q


def path_selection(g, ti, v_s, v_e, df, path, arrival_time, break_search):
    
    
    v_i = v_e
    
    path.append(v_i)
    arrival_time.append(ti)
    
    if v_i == v_s or break_search:
        return {'path': path, 'arrival_time': arrival_time}
    else:
        break_search = True
        stop_find = False
        gi = StepFunction(g[v_i].gi_d, g[v_i].gi_v) 
        
        f_ji = df[(df['xid'] == v_i[0]) & (df['yid'] == v_i[1])]
        f_ji.sort_values(by='hour', ascending=True, inplace=True)        
        f_ji = StepFunction(np.append(f_ji.hour, g[v_i].ta), np.where(f_ji.wind>=15, 1440, 0))
        
        for j in find_neighbors(v_i, limit_x=[1, 548], limit_y=[1, 421]):
            
            if stop_find:
                continue
        
            if j in g:
                
                tj = g[j].gi_d[np.argmin(g[j].gi_v)]
                gj = StepFunction(g[j].gi_d, g[j].gi_v)

                
                find_idx_j = (g[j].gi_d <= (ti-2)) & np.append((g[j].gi_v == (gi(ti) - f_ji(ti))), [False])

                if sum(find_idx_j) > 0:
                    
                #    print g[j].gi_d[g[j].gi_d <= (ti-2)]
                #    print g[j].gi_v[g[j].gi_v == (gi(ti) - f_ji(ti-2))]
                #    print g[j].gi_d[find_idx_j]
                #    print np.max(g[j].gi_d[find_idx_j])
                #    print '----------'
               
                    tj = np.max(g[j].gi_d[find_idx_j])
                    
                    #arrival_time.append(ti-2-tj)
                    
                    v_i = j
                    ti = tj
                    
                    stop_find = True
                    break_search = False
        return path_selection(g, ti, v_s, v_i, df, path, arrival_time, break_search)
        


def translate_num_to_time(x):
    hour = (x+180) // 60
    minutes = (x+180) % 60
    
    if hour < 10:
        hour = '0' + str(hour)
    else:
        hour = str(hour)
        
    if minutes < 10:
        minutes = '0' + str(minutes)
    else:
        minutes = str(minutes)
        
    return hour + ':' + minutes
    
    


def mask(df, key, value):
    return df[df[key] == value].iloc[0]

def create_path(df, df_city, cid, date_id):
    
    start_point = mask(df_city, 'cid', 0)
    end_point = mask(df_city, 'cid', cid)
    
    start_point = (start_point['xid'], start_point['yid'])
    end_point = (end_point['xid'], end_point['yid'])
    
    p_result = compute_minimum_cost(start_point, end_point, 0, 1080, df)
    
    ti = p_result[end_point].gi_d[np.argmin(p_result[end_point].gi_v)]
    
    result_path = path_selection(p_result, ti, start_point, end_point, df, [], [], False)
    
    result_path['arrival_time'].reverse()
    result_path['path'].reverse()
    
    
    
    path_times = np.append(np.diff(result_path['arrival_time'])//2, [1])
    
    path = zip(*result_path['path'])
    
    path = zip(*[np.repeat(path[0], path_times), np.repeat(path[1], path_times)])
    
    
    arrival_time = np.arange(result_path['arrival_time'][0], result_path['arrival_time'][-1]+2, 2)
    
    
    return pd.DataFrame({'Destination City ID': cid, 'Date ID': date_id, 'time': list(map(translate_num_to_time, arrival_time)), 'x-axis': [i for i, j in path], 'y-axis': [j for i, j in path]})
    
