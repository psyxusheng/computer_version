# -*- coding: utf-8 -*-
import time

def timeit(func):
    def run(*args,**kwargs):
        start = time.time()
        ret = func(*args,**kwargs)
        end = time.time()
        print('running %s using %.3f '%(func.__name__,end-start))
        return ret
    return run