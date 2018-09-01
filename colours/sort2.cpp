/* HOW TO COMPILE 'g++-7 hello.cpp -fopenmp' */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
#include <vector>
#include <algorithm>

using namespace std;




void isort(int arr[], int idx[], int NS){

vector<int> x (arr, arr + NS );
vector<int> y (arr, arr + NS );

int ic=0;

    //std::vector<int> y(x.size());
    std::size_t n(0);
    std::generate(std::begin(y), std::end(y), [&]{ return n++; });

    std::sort(  std::begin(y), 
                std::end(y),
                [&](int i1, int i2) { return x[i1] < x[i2]; } );

    for (auto v : y){
        //std::cout << v << ' ';
        idx[ic]=v;
        ic = ic + 1;
        }
   
}











/*
int main() {

int arr[14] = {1,2,3,4,5,6,7,8,5,10,11,12,13,14};
int idx[14] = {0};

int ns = 14;

//isort(arr, idx, n);
isort(arr, idx, ns);


cout << "output from void sort function"<<endl;
for (int i=0;i<ns;i++) cout << idx[i]<<endl;



vector<int> x (arr, arr + 14 );
vector<int> y (arr, arr + 14 );





    //std::vector<int> y(x.size());
    std::size_t n(0);
    std::generate(std::begin(y), std::end(y), [&]{ return n++; });

    std::sort(  std::begin(y), 
                std::end(y),
                [&](int i1, int i2) { return x[i1] < x[i2]; } );

    for (auto v : y)
        std::cout << v << ' ';
    cout << endl;


    
    return 0;
}

*/