/*
 * main.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: jeromethai
 */

#include <iostream>
#include"proj_simplex.h"
#include"isotonic_regression.h"
#include"quadratic_objective.h"

int main( int argc, char ** argv ) {
	test_proj_simplex();
	test_isotonic_regression();
	test_isotonic_regression_2();
	test_quad_obj();
	return 0;
}

