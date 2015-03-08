import ipdb
import time
import logging

import numpy as np

import solvers
from bsls_utils import lsv_operator
import BB, LBFGS, DORE

__author__ = 'cathywu'

class GradientDescent:

    def __init__(self,z0=None,f=None,nabla_f=None,proj=None,method='BB',
                 options=None, A=None, N=None, target=None):
        self.z0 = z0
        self.f = f
        self.nabla_f = nabla_f
        self.proj = proj
        self.method = method

        # DORE only
        self.A = A
        self.N = N
        self.target = target

        if options is None:
            self.options = { 'max_iter': 300000,
                        'verbose': 1,
                        'opt_tol' : 1e-30,
                        'suff_dec': 0.003, # FIXME unused
                        'corrections': 500 } # FIXME unused
        else:
            self.options = options

        self.iters, self.times, self.states = [], [], []
        def log(iter_,state,duration):
            self.iters.append(iter_)
            self.times.append(duration)
            self.states.append(state)
            start = time.time()
            return start
        self.log = log

    def run(self):
        logging.debug('Starting %s solver...' % self.method)
        if self.method == 'LBFGS':
            LBFGS.solve(self.z0+1, self.f, self.nabla_f, solvers.stopping,
                        log=self.log, proj=self.proj, options=self.options)
            logging.debug("Took %s time" % str(np.sum(self.times)))
        elif self.method == 'BB':
            BB.solve(self.z0, self.f, self.nabla_f, solvers.stopping,
                     log=self.log, proj=self.proj, options=self.options)
        elif self.method == 'DORE':
            # setup for DORE
            alpha = 0.99
            lsv = lsv_operator(self.A, self.N)
            logging.info("Largest singular value: %s" % lsv)
            A_dore = self.A*alpha/lsv
            target_dore = self.target*alpha/lsv

            DORE.solve(self.z0, lambda z: A_dore.dot(self.N.dot(z)),
                       lambda b: self.N.T.dot(A_dore.T.dot(b)), target_dore,
                       proj=self.proj, log=self.log, options=self.options,
                       record_every=100)
            A_dore = None
        logging.debug('Stopping %s solver...' % self.method)
        return self.iters, self.times, self.states
