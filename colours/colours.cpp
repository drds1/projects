/* HOW TO COMPILE 'g++-7 hello.cpp -fopenmp' */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
#include <algorithm>
#include <vector>

using namespace std;



// arg sorting routine (I miss numpy.argsort)
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








int main ()
{
  
int sum = 0, value = 0;  
  
// read until end-of-file, calculating a running total of all values read    

  int N,M,B;
  cout << "reading Number of colours, number of columns, number of bricks \n";
  cin >> N >> M >> B;
  int bricks [B][N],score[B];
  int n,i1=0,i2= 0,s=0;
  cout << "N,M,B ="<< N<<" "<<M<<" "<<B;	
  cout << "for each of the '<< B<< 'bricks, reading... \n";
  
  
  // read first enter \n symbol
  cin.get();
  
  for( int i = 0; i < B; i = i + 1 ) { 

  /* read till end-of-file, calculating a running total of all values read
  keep track of score of each brick score[:B]*/
  i1 = 0;
  s = 1;
  cin >> value;
  while (cin.get() != '\n'){
  cout << "value " << value<<" i i1 = "<< i<<" "<< i1 << "\n";
  bricks[i][i1] = value;
  cin >> value;
  
  //std::cout << "Sum is: " << sum << std::endl;
  if (i1 % 2 == 0) {
  s = s + value;
  cout << "scoring i = "<< i<< " i1 == "<<i1<<" value="<<value<<"\n";
  }
  i1 += 1;
  }
  cout << "value " << value<<" i i1 = "<< i<<" "<< i1 << "\n";
  cout << "score " << s << "\n";
  bricks[i][i1] = value;  
  score[i] = s;
  
  //set unused elements to -1
  for (i2=i1+1;i2<N-1;i2 = i2 + 1){
  bricks[i][i2] = -1;
  }
  
  
  cout << "Brick " << i << "\n";
  i2 = 0;
  
  for( int i2 = 0; i2 < N-1; i2 = i2 + 1 ) {
  cout << bricks[i][i2]<< " ";
  }
  cout << " Score = " << score[i];
  cout << "\n \n";
  }
  
 
/* sort the scores so we know which brick gives the highest score.
Prioritise these bricks when filling the grid */
int ibsort[B];
isort(score, ibsort, B )


 
 
// Bricks now read into bricks array with scores in score array 
int iclm,ib, ibnow, bnow,i2,icol,bn,skip_brick,score_save=0;

//array to denote whether column has a particular colour in it
int colfill[N] = {0};

// array to determine if a brick has already been chosen
int bprev[B]={0};

// engage outer loop over columns. Try to get in as many hig scoring bricks as possible
for(iclm=0;iclm<M;iclm=iclm+1){


// inner loop over bricks in ascending order from high value down
for (ib=0;ib<B;ib=ib+1){
if (bprev[ib] == 1) break;

ibnow = ibsort[ib]
skip_brick = 0;

for (i2=0;i2<N;i2=i2+1) bnow[i2]  = bricks[ibnow][i2];

//identify colours in brick and if a space is available in a column assign it else skip to next brick
for (i2=0;i2<N;i2=i2+2) {
bn = bnow[i2];
if (bn == -1 or colfill[bn] == 1) {
skip_brick = 1;
break;
}
colfill[bn] = 1;
}

if (skip_brick == 0){
score_save = score_save + score[ibnow];
bprev[ib] = 1


}

if 



}



}
  
  
}













