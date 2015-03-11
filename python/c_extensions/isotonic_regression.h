/*
 * isotonic_regression.h
 *
 *  Created on: Mar 10, 2015
 *      Author: jeromethai
 */

 #include <iostream> // std::cout

 using std::cout;
using std::endl;

void isotonic_regression(double *y, int start, int end) {
	// Do isotonic regression from start to end (end not included)
    double numerator, previous;
    int i, j, k, pooled, denominator;
    // initialize the weights to 1
    int weight[end-start];
    for (i = 0; i < end-start; i++) weight[i] = 1;
    while (1) {
    	// repeat until there are no more adjacent violators.
    	i = start;
    	pooled = 0;
    	while (i < end) {
    		k = i + weight[i-start];
    		previous = y[i];
            while (k < end && y[k] <= previous) {
            	previous = y[k];
            	k += weight[k-start];
            }
            if (y[i] != previous) {
            	// y[i:k + 1] is a decreasing subsequence, so
                // replace each point in the subsequence with the
                // weighted average of the subsequence.
                numerator = 0.0;
                denominator = 0;
                j = i;
                while (j < k) {
                	numerator += y[j] * weight[j-start];
                    denominator += weight[j-start];
                    j += weight[j-start];
                }
                y[i] = numerator / denominator;
                weight[i-start] = denominator;
                pooled = 1;
            }
            i = k;
    	}
    	// Check for convergence
    	if (pooled == 0) break;
    }
    for (i = start; i < end; i++) {
    	k = i + weight[i-start];
    	for (j = i + 1; j < k; j++) y[j] = y[i];
    }

}


void isotonic_regression_multi(double *y, int *blocks, int numblocks, int n) {
	// Do multiple iso tonic regressions where blocks is an array of integers 
	// that contains the first index of each block and n the length of the array
	for (int i = 0; i < numblocks-1; i++) {
		isotonic_regression(y, blocks[i], blocks[i+1]);
	}
	isotonic_regression(y, blocks[numblocks-1], n);
}


int test_isotonic_regression() {
	double doubleArray[] = {4., 5., 1., 6., 8., 7.};
    int blocks[] = {0};
    int blocks2[] = {0, 2, 4};

    isotonic_regression(doubleArray, 0, 6);
    cout << "Projected block-vector is this." << endl;
    for (size_t i = 0; i != 6; ++i)
        cout << doubleArray[i] << " "; // should get 3.3, 3.3, 3.3, 6. 7.5, 7.5
        cout << endl;

    double doubleArray2[] = {4., 5., 1., 6., 8., 7.};
    isotonic_regression_multi(doubleArray2, blocks, 1, 6);
    cout << "Projected block-vector is this." << endl;
    for (size_t i = 0; i != 6; ++i)
        cout << doubleArray2[i] << " "; // should get 3.3, 3.3, 3.3, 6. 7.5, 7.5
        cout << endl;

    double doubleArray3[] = {4., 5., 1., 6., 8., 7.};
    isotonic_regression_multi(doubleArray3, blocks2, 3, 6);
    cout << "Projected block-vector is this." << endl;
    for (size_t i = 0; i != 6; ++i)
        cout << doubleArray3[i] << " "; // should get 3.3, 3.3, 3.3, 6. 7.5, 7.5
        cout << endl;

    return 0;
}