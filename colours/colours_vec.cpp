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






/* main body of code below (above void function used for sorting further down) */

int main ()
{
  
int  value = 0;  
  
  int N,M,B,N2;
  cin >> N >> M >> B;
  N2 = 2*N;
  vector< vector<int> > bricks(B, vector<int> (N2, -1)); 
  vector<int> ibsort(B),score(B);
  int i1=0,i2= 0,s=0;

  
  // read first enter \n symbol
  cin.get();
  
  for( int i = 0; i < B; i = i + 1 ) { 

  /* read till end-of-file, calculating a running total of all values read
  keep track of score of each brick score[:B]*/
  i1 = 0;
  s = 0;
  cin >> value;
  while (cin.get() != '\n' and i1 < N2){
  
  if (i == B-1){
  if (cin.fail()) break;
  }
  

  bricks[i][i1] = value;
  cin >> value;
  

  if (i1 % 2 == 0) {
  s = s + value;
  }
  i1 += 1;
  }

  bricks[i][i1] = value;  
  score[i] = s;
  
  //set unused elements to -1
  for (i2=i1+1;i2<N2;i2 = i2 + 1){
  bricks[i][i2] = -1;
  }
  
  
 
  i2 = 0;
  
    for( int i2 = 0; i2 < N2-1; i2 = i2 + 1 ) {
  if (bricks[i][i2] == -1) break;

  }

  }
 
 
 /* input read into bricks[:B][:2N] vector scores into score[:B} vector */
  
 
 

/* check for each multicoloured brick, a higher score cant be achieved 
using a collection of smaller bricks with collectively the same colours
by adding fake scores to the correct bricks with fewer colours to trick the sorting
algorithm */



 
 
 
 
/* sort the scores so we know which brick gives the highest score.
Prioritise these bricks when filling the grid */
int x=0;
iota(ibsort.begin(),ibsort.end(),x++); //Initializing sorting code (void function at top)
sort( ibsort.begin(),ibsort.end(), [&](int i,int j){return score[i]<score[j];} );





 
 
// Bricks now read into bricks vector with scores in score vector
int iclm,ib, ibnow,bn,skip_brick,icolsum,score_save=0;

//vectors to denote whether column has a particular colour in it (colfill)
// and bprev vector to identify if a brick has already been selected for previous columns
vector<int> colfill(N,0),bnow(N2,0),bprev(B,0);
vector< vector<int> > colsave(M, vector<int> (N, -1));




// engage outer loop over columns. Try to get in as many hig scoring bricks as possible
for(iclm=0;iclm<M;iclm=iclm+1){
icolsum=0;

// reset the colfill array to zeros for each new column
for (int ic=0;ic<N;ic++) colfill[ic] = 0;


// inner loop over bricks in ascending order from high value down
for (ib=0;ib<B;ib=ib+1){
ibnow = ibsort[B-1-ib];


// if a brick has already been assigned, save time by skipping to the next one.
if (bprev[ibnow] == 1) {
continue;
}


skip_brick = 0;


for (i2=0;i2<N2;i2=i2+1) {
bnow[i2]  = bricks[ibnow][i2];
}

//identify colours in brick and, if a space is available in a column, assign it else skip to next brick
for (i2=0;i2<N2;i2=i2+2) {
bn = bnow[i2];
// if the colour is already filled by another brick skip and move on
if (colfill[bn] == 1) {
skip_brick = 1;
break;
}
if (bn == -1) break;
colfill[bn] = 1;



}


/* if the brick checks out put it in the column and record the score 
also record that the brick has already been chosen to save time in future loops */
if (skip_brick == 1) continue;


score_save = score_save + score[ibnow];
bprev[ibnow] = 1;
colsave[iclm][icolsum]=ibnow;
icolsum = icolsum + 1;

//stop once the end of the brick info is reached for the current brick
for (int ic = 0;ic < N;ic++){
if (bnow[ic] == -1) break;
}



} //end ib loop over bricks

//set empty spaces in column to zero
for (i2 = icolsum;i2<N;i2=i2+1) colsave[iclm][i2] = -1;

} //end iclm loop over columns



cout << score_save << "\n";
for (iclm = 0; iclm < M ; iclm = iclm + 1){
for (i2=0;i2<N;i2=i2+1){
if (colsave[iclm][i2] == -1) break;
cout << colsave[iclm][i2] << " ";
}
cout << "\n";
}

} // end program








