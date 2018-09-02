/* Script to organize blocks */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
#include <algorithm>
#include <vector>

using namespace std;


template <class ForwardIterator, class T>
  void iota (ForwardIterator first, ForwardIterator last, T val)
{
  while (first!=last) {
    *first = val;
    ++first;
    ++val;
  }
}

int main(){

int N;

N=3;
//Assume A is a given vector with N elements
vector<int> V(N),A(N);

A[0]=3;
A[1]=4;
A[2]=1;

int x=0;
iota(V.begin(),V.end(),x++); //Initializing
sort( V.begin(),V.end(), [&](int i,int j){return A[i]<A[j];} );

for (int i=0;i<N;i=i+1) cout<<V[i]<< " ";
cout << endl;

}