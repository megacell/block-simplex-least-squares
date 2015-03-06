Convex optimization for traffic assignment
==========================================

Setup
-----
To run the MATLAB implementation, see [MATLAB setup](#matlab-setup)

Python dependencies:

    sudo easy_install pip
    pip install -r requirements.txt

Also needed is scipy. If you find some missing dependencies, please add them here.

To build the simplex projection c extension:
1. `cd` into `python/c_extensions`
2. run `python2 setup.py build_ext --inplace`

Running via Python
-------------------
Run the python implementation from the `traffic-estimation/python` directory.

To run the main test, see these examples:
```
cd ~/traffic-estimation/python
python main.py --file route_assignment_matrices_ntt.mat --log=DEBUG --solver LBFGS
python main.py --file route_assignment_matrices_ntt.mat --log=DEBUG --solver BB
python main.py --file route_assignment_matrices_ntt.mat --log=DEBUG --solver DORE
```
If the dataset you want to run is not in the data directory, symlink it in
from the main dataset.

To run 3-fold cross validation test:
```
python CrossValidation.py --log=DEBUG
```

Running ISTTT
-------------
After generating the set of matrices run:
```
python ISTTT.py --log=DEBUG --solver BB
```

MATLAB setup
------------
<a name="matlab-setup"></a>
MATLAB dependencies (must be run every time MATLAB is started):

    setup.m


c_proj_l1ball setup
-------------------

Instructions to bind the c-implementation of proj_l1ball to Python

    cd python/c_proj_l1ball
    python setup.py build_ext --inplace 

check if it created an executable 'proj_l1ball_c.so' and 'proj_l1ball_c.cpp'
now run this to test the implementation

    python call_cpp.py

works if no assertion error

Running via MATLAB
-------------------
Run `main.m`.

References
--------
Mark Schmidt's [L1General](http://www.di.ens.fr/~mschmidt/Software/L1General.html), a set of Matlab routines for solving L1-regularization problems. 
