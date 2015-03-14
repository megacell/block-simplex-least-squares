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


CPLEX Setup
-----------

First download CPLEX for your OS from the IBM website (you'll need to sign up
for an academic initiative account first). It would be a file named
`CPLEX_xxxxxxxxxx.bin`.

To install, run (using `sudo` if necessary):

```
chmod +x CPLEX_xxxxxxxxxx.bin
./CPLEX_xxxxxxxxxx.bin
```

Note the installation directory, then to install the python bindings (using
`sudo` if necessary, for OSX users you might need to run `sudo -s` first before
these steps):

```
cd <installation-directory>/ILOG/CPLEX_Studio1261/cplex/python/2.7/x86-64_linux
python setup.py install
```

Install openopt (documentation here: http://openopt.org/cplex):
```
pip install openopt
```

To see an example on how to use CPLEX, look at `tests/fast/test_cplex.py`


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


c_extensions setup
-----------------
To bind Cythonic extensions to Python

    cd python/c_extensions
    python setup.py build_ext --inplace

check if it created an executable 'c_extensions.so' and 'c_extensions.cpp'

Running via MATLAB
-------------------
Run `main.m`.

References
--------
Mark Schmidt's [L1General](http://www.di.ens.fr/~mschmidt/Software/L1General.html), a set of Matlab routines for solving L1-regularization problems.
