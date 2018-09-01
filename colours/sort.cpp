/* HOW TO COMPILE 'g++-7 hello.cpp -fopenmp' */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
#include <vector>
#include <algorithm>

using namespace std;



void isort(int myints[],int idout[],int n ){

int i2=0, myints_in[n];
for (int i = 0; i < n; i++) myints_in[i] = myints[i];
for (int i = 0; i < n; i++)
   myints_in[i] = myints_in[i]*(1 << 2*n) + i;
std::vector<int> myvector(myints_in, myints_in+n);
sort(myvector.begin(), myvector.begin()+n, std::less<int>());
for (std::vector<int>::iterator it = myvector.begin(); it != myvector.end(); ++it){
   idout[i2] = (*it)%(1 << 2*n);
   i2 = i2 + 1;}

}



/* sorting code test
int main(){
int n = 8,i1=0;
int myints[] = {26,71,12,45,26,12,53,33};
int idsort[] = {0,0,0,0,0,0,0,0};

isort(myints,idsort,n);
cout << "\n";
for (i1=0;i1<n;i1=i1+1) cout << idsort[i1] << " ";
cout << "\n";
for (i1=0;i1<n;i1=i1+1) cout << myints[i1] << " ";
cout << "\n";
for (i1=0;i1<n;i1=i1+1) cout << myints[idsort[i1]] << " ";
cout << "\n";

}

*/