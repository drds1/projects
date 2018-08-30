!!gfortran calc_tennis.f90 -o calct
!!./calct

!David Starkey - StAndrews 13/2/17 ds207@st-andrews.ac.uk
!!type the top 2 lines (-!!) into the terminal to run on mac
!!calculate directly the probability of a given score in tennis
!! the monte carlo method is less precise (only good to ~3dp)
!! subroutine dependencies all in this file





program calc_tennis

double precision p_point,probout,p_gwin,psout,pnorm
double precision sum,sum2
double precision,allocatable:: posterior(:,:)
!defaults below. Alter 'calc_tennis.par' file to change ipnuts
p_point = 0.52
i1=6
i2=3
nsum = 1000
maxgame = 7


allocate(posterior(maxgame,maxgame))

open(unit = 1,file ='calc_tennis.par')
read(1,*) i1,i2
read(1,*) p_point
close(1)

write(*,*) 'probability of wining point',p_point



!! calculate explicitly the probability of winning deuce using geometric series to infinity
call prob_win_deuce(p_point,probout)
!!


!!! calculate the probability of winning the game
call prob_win_game(p_point,nsum,p_gwin)
write(*,*) 'probability of winning game',p_gwin
!!!


!!! calculate the probability of the score being a certain value
call probscore(i1,i2,p_gwin,psout)
write(*,*) 'prob of score',i1,i2,'=',psout






i1save = i1
i2save = i2
write(*,*)''
! show all scores winning scores
i1=6
sum2 = 0.d0
do i2 = 1,4
sum = 0.d0
call probscore(i1,i2,p_gwin,sum)
write(*,*) 'win prob of score',i1,i2,'=',sum
sum2 = sum2 + sum
enddo
i1 = 7
i2 = 5
sum = 0.d0
call probscore(i1,i2,p_gwin,sum)
write(*,*) 'win prob of score',i1,i2,'=',sum
sum2 = sum2 + sum
i1 = 7 
i2 = 6
sum = 0.d0
call probscore(i1,i2,p_gwin,sum)
write(*,*) 'tie break win prob of score',i1,i2,'=',sum
sum2 = sum2 +sum
!
!losing scores
i2 = 6
do i1 = 1,4
sum = 0.d0
call probscore(i1,i2,p_gwin,sum)
write(*,*) 'lose prob of score',i1,i2,'=',sum
sum2 = sum2 + sum
enddo
i2 = 7
i1 = 5
sum = 0.d0
call probscore(i1,i2,p_gwin,sum)
write(*,*) 'lose prob of score',i1,i2,'=',sum
sum2 = sum2 + sum
i2 = 7 
i1 = 6
sum = 0.d0
call probscore(i1,i2,p_gwin,sum)
write(*,*) 'tie break lose prob of score',i1,i2,'=',sum
sum2 = sum2 +sum

!!!




!!write the output as a 2D probability distribution for python plotting

open(unit = 1,file = 'mc_tennis_posterior.txt')
do id1 = 1,maxgame
do id2 = 1,maxgame
call probscore(id1,id2,p_gwin,posterior(id1,id2))
write(1,*) id1,id2,posterior(id1,id2)
enddo
enddo
close(1)

end program












!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! calculate the probability of a certain score (flip if need to lose)
!INPUT: i1,i2 (Djokovic, Murray score [integer])
!.....: pg1w (probability of djokovic winning game --> see prob_win_game routine [dp]) 

!OUTPUT: psout (probability of score i1,i2 [dp])

subroutine probscore(i1,i2,pg1w,psout)
double precision, intent(in):: pg1w
integer, intent(in):: i1, i2
double precision,intent(out):: psout




if (i1 .gt. i2) then
 call probscore_2(i1,i2,pg1w,psout)
else
 call probscore_2(i2,i1,1.-pg1w,psout) ! i think this is ok e.g saying prob of winning 6,4 is like probability of playing a losding game 6,4 with game win prob changed to game lose prob
endif
end subroutine


!!! calculate the probability of a certain score (same as probscore 1 but flips probabilities if losing)
!INPUT: i1,i2 (Djokovic, Murray score [integer])
!.....: pg1w (probability of djokovic winning game --> see prob_win_game routine [dp]) 

!OUTPUT: psout (probability of score i1,i2 [dp])
subroutine probscore_2(i1,i2,pg1w,psout)
double precision, intent(in):: pg1w
integer, intent(in):: i1, i2
double precision,intent(out):: psout
double precision:: fact,factbit,pg1l
!!winning scores

!p6n

pg1l =1.-pg1w
!
if ((i1 .eq. 6) .and. (i2 .le. 4))  then ! for scores of 6,1:4
 factbit = fact(i1+i2-1)/(fact(i1-1)*fact(i2))
 psout   = pg1w**i1 * pg1l**i2 * factbit
else if (i1 .eq. 7 .and. i2 .eq. 5) then !7,5 score
 factbit = fact(10)/(fact(5)*fact(5))
 psout   = pg1w**i1 * pg1l**i2 *factbit
else if (i1 .eq. 7 .and. i2 .eq. 6) then !tie break
 factbit = fact(10)/(fact(5)*fact(5))
 psout   = pg1w**i1 * pg1l**i2 *factbit*2
else 
 !write(*,*) 'please enter valid score'
 psout = 0
 !stop
endif
end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






!!!!!!! subroutine to calculate the probability of winning a game !!!!!!!!!
!nsum how many deuces do i sum over before negelcting (try 1000)
!INPUT: p_point (Djokovic point win prob [dp])
!.....: nsum (sum over all the possibilities to win a deuce - probs get smaller with higher nsum so stop after many iterations)
!.....: Update 13/2/17 nsum unused. The win deuce prob can be cast as geometric sum to infinity [or using Bayes' theorem]. Can calculate exactly

!OUTPUT: p_win (probability of winning game [dp])

subroutine prob_win_game(p_point,nsum,p_win)
double precision,intent(in):: p_point
integer,intent(in):: nsum
double precision,intent(out):: p_win
double precision sum,pnow,p_notpoint,prob_bd,p_g2d

p_notpoint = 1. - p_point

! can do the adding or use geometric series idea
!sum = 0.d0
!do it = 0,nsum
!call prob_nthdeuce_win(p_point,it,pnow)
!sum = sum + pnow
!enddo

!uses geometric sum to consider winning on 1st deuce up to infinity
!need to add that on to probability of winning before deuce prob_bd
!either win before deuce, or get2deuce and win deuce
prob_bd = p_point**4 * (1. + 4*p_notpoint + 10*p_notpoint**2)

!calculate the probability of getting to deuce
call prob_get2deuce(p_point,p_g2d)

call prob_win_deuce(p_point,sum)
p_win = p_g2d*sum + prob_bd

end subroutine
!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!











!!!!!!!!!!!! subroutine to work out the prob of winning game after n'th deuce (3-3) !!!!!!!!!!!
!INPUT: p_point (Djokovic point win prob [dp])
!.....: n (win after n'th deuce [int])

!OUTPUT: prob winning deuce after nth return to deuce (0 = win before deuce)


!subroutine prob_nthdeuce_win(p_point,n,probout)
!integer,intent(in):: n
!double precision,intent(in)::p_point
!double precision,intent(out):: probout
!double precision p_notpoint, p_g2d
!p_notpoint = 1. - p_point
!
!!calculate the probability of getting to deuce
!call prob_get2deuce(p_point,p_g2d)
!
!!calculate prob of winning on deuce
!p_wind = p_point*p_point
!
!!calculate probability of returning to deuce
!p_b2d  = 2*p_point*p_notpoint
!
!
!
!if (n == 0) then !win before deuce
! probout = p_point**4 * (1. + 4*p_notpoint + 10*p_notpoint**2)
!else if (n ==1) then
! probout = p_g2d*p_wind
!else
! probout = p_g2d * p_b2d**(n-1) * p_wind
!endif
!
!end subroutine 
!!!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! 


!! subroutine to explicitly work out probability of winning on deuce (3-3)
!!uses geometric series sum to infinity
!sum_1^infty (2*p_point*(1 - p_point))^(n-1) *p_point^2
!in general for geometric series 
!sum_[n=1]^[infinity] = a r^(n-1) = a_1 / (1 - r) 

!INPUT: p_point (Djokovic point win prob [dp])
!OUTPUT: probout (probability of winning deuce [dp])
subroutine prob_win_deuce(p_point,probout)
double precision,intent(in):: p_point
double precision,intent(out):: probout

probout = p_point*p_point / (1. - 2*p_point*(1.-p_point))

end subroutine







!!!! subroutine to work out the prob of getting to deuce !!!!
!INPUT: p_point (Djokovic point win prob [dp])
!OUTPUT: probout (probability of game reaching deuce (3-3))
subroutine prob_get2deuce(p_point,probout)
double precision,intent(in):: p_point
double precision,intent(out):: probout

probout = 20 * p_point**3 * (1.-p_point)**3

end subroutine
!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! 












!!!!!!!!!!! subroutine to calculate factorial !!!!!!!!!!! !!!!!!!!!!! 
function fact(ifact)
integer,intent(in):: ifact
double precision dfact, fact
dfact = 1
do i = 1,ifact
dfact = dfact*i
enddo

fact = dfact
end function
!!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!! 
