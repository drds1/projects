/* HOW TO COMPILE 'g++-7 hello.cpp -fopenmp' */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
#include <vector>
#include <algorithm>

using namespace std;




//convert arrays to vectors
int main (){
int i,n=3;
int arr[3] = {1, 2, 3};
//std::vector<int> v(x, x + n / n);
vector<int> v (arr, arr + sizeof(arr) / sizeof(arr[0]) );

for (i=0; i<n;i++) cout << v[i] << " ";
cout << endl;
}