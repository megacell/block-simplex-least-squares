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

Set up the pre-commit hook (which automatically runs the fast unit tests):

      ln -s ../../pre-commit.sh .git/hooks/pre-commit

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


proj_simplex_c and isotonic_regression_c setups
-----------------    
To bind Cythonic proj_simplex_c to Python

    cd python/proj_simplex
    python setup.py build_ext --inplace 

check if it created an executable 'proj_simplex_c.so' and 'proj_simplex_c.cpp'
now run this to test the implementation

    python call_proj_simplex_c.py

To set up isotonic_regression_c adapted from:
http://tullo.ch/articles/speeding-up-isotonic-regression/

    cd ../isotonic_regression
    python setup.py build_ext --inplace
    python call_isotonic_regression_c.py

Running via MATLAB
-------------------
Run `main.m`.

References
--------
Mark Schmidt's [L1General](http://www.di.ens.fr/~mschmidt/Software/L1General.html), a set of Matlab routines for solving L1-regularization problems. 
