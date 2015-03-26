/*
 * quadratic_objective.h
 *
 *  Created on: Mar 10, 2015
 *      Author: jeromethai
 */

#include <iostream> // std::cout
//#include <cstddef>  
#include <algorithm> // std::sort

using std::cout;
using std::endl;

double quad_obj(double *x, double *Q, double *c, double*g, int n) {
	// Compute quadratic objective .5*x'*Q*x + c*x
	double f = 0;
	int k;
	for (int i = 0; i < n; i++) {
		g[i] = c[i];
		k = i*n;
		for (int j = 0; j < n; j++) g[i] += Q[k+j] * x[j];
        f += 0.5 * (g[i] + c[i]) * x[i];
	}
	return f;
}


double line_search(double *x, double f, double *g, 
	               double *x_new, double f_new, double *g_new,
	               double *Q, double *c, int n) {
	// Do backtracking line search
	double t = 1, suffDec = 1e-4, upper_line = f, progTol = 1e-8, max;
	int i, j, k;
	// Compute initial upper_line
    for (i = 0; i < n; i++) upper_line += suffDec * g[i] * (x_new[i] - x[i]);
    // main loop
	while (f_new > upper_line) {
		t *= .5;
		// compute norm_inf of x - x_new
		max = 0.0;
		for (i = 0; i < n; i++) {
			if (x_new[i] - x[i] > max) max = x_new[i] - x[i];
            if (x[i] - x_new[i] > max) max = x[i] - x_new[i];
		}
		// Check whether step has become too small
        if (t * max < progTol) {
            t = 0.0;
            for (i = 0; i < n; i++) x_new[i] = x[i]; g_new[i] = g[i];
            f_new = f;
            break;
        }
        // Update point, objective value, gradient
        for (i = 0; i < n; i++) x_new[i] = x[i] + t * (x_new[i] - x[i]);
        f_new = quad_obj(x_new, Q, c, g_new, n);
        // Update upper_line
	    for (i = 0; i < n; i++) upper_line += suffDec * g[i] * (x_new[i] - x[i]);

	}
	return f_new;
}


int test_quad_obj() {
    cout << "Test quad_obj." << endl;
    double Q[] = {4., 1, 1, 2.};
    double x[] = {0.25, 0.75};
    double c[] = {1., 1.};
    double g[] = {0., 0.};
    int n = 2;
    double f = quad_obj(x, Q, c, g, n);
    cout << "Objective value: " << f << endl;

    double x2[] = {0.26, 0.74};
    double f2 = 1.8752;
    double g2[] = {2.78, 2.74};
    double x_new[] = {0., 1.};
    double f_new = 2.0;
    double g_new[] = {2., 3.};
    f = line_search(x2, f2, g2, x_new, f_new, g_new, Q, c, 2);
    cout << "Objective value: " << f << endl;

    return 0;
}