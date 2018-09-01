/* HOW TO COMPILE 'g++-7 hello.cpp -fopenmp' */
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>                           /* math functions */
using namespace std;

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
  
  
  
}













