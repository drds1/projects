/* HOW TO COMPILE 'g++-7 hello.cpp -fopenmp' */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
#include <algorithm>
#include <vector>

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







int main ()
{
  
int sum = 0, value = 0;  
  
// read until end-of-file, calculating a running total of all values read    

  int N,M,B,N2;
  cout << "reading Number of colours, number of columns, number of bricks \n";
  cin >> N >> M >> B;
  N2 = 2*N;
  int bricks [B][N2],score[B];
  int n,i1=0,i2= 0,s=0;
  cout << "\n N,M,B ="<< N<<" "<<M<<" "<<B;	
  cout << "\n for each of the '<< B<< 'bricks, reading... \n";
  
  
  // read first enter \n symbol
  cin.get();
  
  for( int i = 0; i < B; i = i + 1 ) { 

  /* read till end-of-file, calculating a running total of all values read
  keep track of score of each brick score[:B]*/
  i1 = 0;
  s = 1;
  cin >> value;
  while (cin.get() != '\n'){
  //cout << "value " << value<<" i i1 = "<< i<<" "<< i1 << "\n";
  bricks[i][i1] = value;
  cin >> value;
  
  //std::cout << "Sum is: " << sum << std::endl;
  if (i1 % 2 == 0) {
  s = s + value;
  //cout << "scoring i = "<< i<< " i1 == "<<i1<<" value="<<value<<"\n";
  }
  i1 += 1;
  }
  //cout << "value " << value<<" i i1 = "<< i<<" "<< i1 << "\n";
  //cout << "score " << s << "\n";
  bricks[i][i1] = value;  
  score[i] = s;
  
  //set unused elements to -1
  for (i2=i1+1;i2<N2;i2 = i2 + 1){
  bricks[i][i2] = -1;
  }
  
  
  cout << "Brick " << i << "\n";
  i2 = 0;
  
  for( int i2 = 0; i2 < N2-1; i2 = i2 + 1 ) {
  if (bricks[i][i2] == -1) break;
  cout << bricks[i][i2]<< " ";
  }
  cout << " Score = " << score[i];
  cout << "\n \n";
  }
  
 
/* sort the scores so we know which brick gives the highest score.
Prioritise these bricks when filling the grid */
int ibsort[B]={0};

isort(score, ibsort, B );


cout << B<<" sorted \n";
for (i2=0;i2<B;i2=i2+1)cout << i2<<" "<<score[i2]<<" "<<ibsort[i2]<< endl;

 
 
// Bricks now read into bricks array with scores in score array 
int iclm,ib, ibnow,icol,bn,skip_brick,icolsum,score_save=0;

//array to denote whether column has a particular colour in it
int colfill[N] = {0},colsave[M][N],bnow[N2];

// array to determine if a brick has already been chosen
int bprev[B]={0};

// engage outer loop over columns. Try to get in as many hig scoring bricks as possible
cout << M<<" here \n"<< endl;
for(iclm=0;iclm<M;iclm=iclm+1){
icolsum=0;

cout << "icolsum,iclm "<< icolsum << " "<< iclm << " "<< M << endl; 
// inner loop over bricks in ascending order from high value down
for (ib=0;ib<B;ib=ib+1){
if (bprev[ib] == 1) break;

ibnow = ibsort[B-1-ib];
skip_brick = 0;

cout << " about to enter i2 loop ibnow "<<ibnow<<endl;
for (i2=0;i2<N2;i2=i2+1) {
bnow[i2]  = bricks[ibnow][i2];
cout << "i2 N2 bnow "<< i2<< " " << N2 << " " << bnow[i2] << endl;
}

//identify colours in brick and, if a space is available in a column, assign it else skip to next brick
for (i2=0;i2<N2;i2=i2+2) {
bn = bnow[i2];
if (colfill[bn] == 1) {
skip_brick = 1;
break;
if (bn == -1) break;
}
colfill[bn] = 1;
}


/* if the brick checks out put it in the column and record the score 
also record that the brick has already been chosen to save time in future loops */
if (skip_brick == 0){
score_save = score_save + score[ibnow];
bprev[ib] = 1;
colsave[iclm][icolsum]=ibnow;
icolsum = icolsum + 1;
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













