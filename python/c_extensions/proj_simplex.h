/*
 * proj_simplex.h
 *
 *  Created on: Nov 27, 2014
 *      Author: jeromethai
 */

#include <iostream> // std::cout
//#include <cstddef>  
#include <algorithm> // std::sort

using std::cout;
using std::endl;

const int SIZE = 7;

void proj_simplex(double *y, int start, int end) {
/*
Projects the subvector between start and end on the simplex
*/
    double u[end-start];
	std::copy(y+start, y+end, u);
    std::sort(u, u+end-start, std::greater<double>());
    double sum = u[0];
    double lambda = 1.-sum;
    double tmp;
    int i;
    for (i = 1; i < end-start; i++) {
        sum += u[i];
        tmp = (1.-sum) / ((double)i + 1.);
        if (u[i] + tmp > 0) lambda = tmp;
    }
    for (i = start; i < end; i++) y[i] = std::max(lambda + y[i], 0.);
}


void proj_multi_simplex(double *y, int *blocks, int numblocks, int n) {
/*
Do multiple projections where blocks is an array of integers that
constains the first index of each block and n the length of the array
*/
    int i;
    for (i = 0; i < numblocks-1; i++) {
        proj_simplex(y, blocks[i], blocks[i+1]);
    }
    proj_simplex(y, blocks[numblocks-1], n);
}


void proj_multi_ball(double *y, int *blocks, int numblocks, int n) {
/*
Do multiple projections where blocks is an array of integers that
constains the first index of each block and n the length of the array
*/ 
double sum;
    for (int i=0; i < numblocks-1; i++) {
        sum = 0.0;
        for (int j=blocks[i]; j < blocks[i+1]; j++) {
            if (y[j] < 0.0) 
                y[j] = 0.0;
            else
                sum += y[j];
        }
        if (sum > 1.0) proj_simplex(y, blocks[i], blocks[i+1]);
    }
    sum = 0.0;
    for (int j=blocks[numblocks-1]; j < n; j++) {
        if (y[j] < 0.0)
            y[j] = 0.0;
        else
            sum += y[j];
    }
    if (sum > 1.0) proj_simplex(y, blocks[numblocks-1], n);
}


int test_proj_simplex() {

    double doubleArray[] = {5.352, 3.23, 32.78, -1.234, 1.7, 104., 53.};
    int blocks[] = {0, 2, 4};

    proj_multi_simplex(doubleArray, blocks, 3, 7);
    cout << "Projected block-vector is this." << endl;
    for (size_t i = 0; i != SIZE; ++i)
        cout << doubleArray[i] << " ";
        cout << endl;

    double doubleArray2[] = {0.234, 0.5, 1.3, -1.234, 1.7, 104., 53.};
    proj_multi_ball(doubleArray2, blocks, 3, 7);
    cout << "Projected block-vector is this." << endl;
    for (size_t i = 0; i != SIZE; ++i)
        cout << doubleArray2[i] << " ";
        cout << endl;

    return 0;
}
