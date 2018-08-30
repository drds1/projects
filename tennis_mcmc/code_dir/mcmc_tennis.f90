! gfortran -fbounds-check -fno-backtrace -Wuninitialized mcmc_tennis.f90 ../ran_new2.for
! gfortran mcmc_tennis.f90 ../ran_new2.for
!! define mcmcm tennis
!! need uniform random number generator
!tie break game rather than tie break point
program test_mc_tennis

!defaults
probwin = 0.50
nmatch = 100000
isd = 6
ism = 3
rules=1

open(unit=1,file='mc_tennis_in.par')
read(1,*) nmatch
read(1,*) probwin
read(1,*) isd,ism
read(1,*) rules
close(1)


call mc_tennis(nmatch,probwin,isd,ism,rules,probscore)
write(*,*)'ruleset',rules
write(*,*) 'probpoint',probwin
write(*,*) 'prob of score',isd,ism,'is',probscore
end program















!subroutine to run nmatch tennis matches with djokovic prob of point probwin,
!output: prob of the final score being isd:ism
subroutine mc_tennis(nmatch,probwin,isd,ism,rules,probscore)
integer,intent(in):: nmatch,isd,ism
real,intent(in):: probwin,rules
real,intent(out):: probscore
integer igwin_save(nmatch),iglos_save(nmatch)

info = 0
info_out = 1

!write(*,*)'bug!! djokovic can lose individual games but always wins the match!'
!write(*,*) 'fix!!'
!stop

call system_clock(iseed)

idjwin = 0

idcorscore = 0
do i1 = 1,nmatch
setwin = 0
setlose = 0
!!set loop
igwin = 0
iglose = 0
setdone = 0

idgame = 0
do while (setdone == 0)


if (rules ==1) then
 
 
 if (((igwin .ge. 6) .and. (igwin - iglose .ge. 2)) &
 .or. ((igwin .eq. 7) .and. (iglose .eq. 6))) then!set won
 setwin = 1
 setdone = 1
 if ((igwin .eq. isd) .and. (iglose .eq. ism)) idcorscore = idcorscore + 1
 igwin_save(i1) = igwin
 iglos_save(i1) = iglose
 exit
 endif
 
 if (((iglose .ge. 6) .and. (iglose - igwin .ge. 2)) &
 .or. ((iglose .eq. 7) .and. (igwin .eq. 6))) then!set lost
 setlose = 1
 setdone = 1
 if ((igwin .eq. isd) .and. (iglose .eq. ism)) idcorscore = idcorscore + 1
 igwin_save(i1) = igwin
 iglos_save(i1) = iglose
 exit
 endif


else if (rules == 2) then

 if (((igwin .ge. 6) .and. (igwin - iglose .ge. 2)) )then!&
 !.or. ((igwin .eq. 7) .and. (iglose .eq. 6))) then!set won
 setwin = 1
 setdone = 1
 if ((igwin .eq. isd) .and. (iglose .eq. ism)) idcorscore = idcorscore + 1
 igwin_save(i1) = igwin
 iglos_save(i1) = iglose
 exit
 endif 

 if (((iglose .ge. 6) .and. (iglose - igwin .ge. 2)) )then!&
 !.or. ((iglose .eq. 7) .and. (igwin .eq. 6))) then!set lost
 setlose = 1
 setdone = 1
 if ((igwin .eq. isd) .and. (iglose .eq. ism)) idcorscore = idcorscore + 1
 igwin_save(i1) = igwin
 iglos_save(i1) = iglose
 exit
 endif
 if ((igwin .eq. 6) .and. (iglose .eq. 6)) then !tie break for the set check if it is tie break point or tie break game

 if (ran3(iseed) .le. probwin) then
 setwin = 1
 igwin = igwin+1
 else
 setlose = 1
 iglose = iglose + 1
 endif
 if ((igwin .eq. isd) .and. (iglose .eq. ism)) idcorscore = idcorscore + 1
 setdone = 1
 exit
 endif
endif
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!points within a game !put this in set loop
ipwin = 0
iplose = 0
gamedone = 0
!idxcheck=0
gwin = 0
glose = 0
do while (gamedone ==0.0)
 if (ran3(iseed) .le. probwin) then!djokovic wins
 ipwin = ipwin+1
 else !djokovic loses
 iplose = iplose+1
 endif
 
 if ((ipwin .ge. 4) .and. (ipwin-iplose .ge. 2)) then
 gwin = 1
 igwin = igwin + 1
 exit
 !write(*,*) 'game',idgame,'djokovic win',ipwin,iplose
 endif
 
 if ((iplose .ge. 4) .and. (iplose-ipwin .ge. 2)) then
 glose = 1
 iglose = iglose + 1
 exit
 !write(*,*) 'game',idgame,'djokovic lose',ipwin,iplose
 endif
 !idxcheck=idxcheck+1
 !write(*,*) gwin,glose,'check',igwin,iglose,ipwin,iplose,gamedone,idxcheck
 if ((gwin .gt. 0) .or. (glose .gt. 0)) gamedone = 1
enddo
idgame = idgame + 1
!read(*,*)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!end of game  loop


enddo !end do while set

if (setwin ==1) then!dokovic win
 idjwin = idjwin + 1
 if (info ==1)write(*,*) 'end of match',i1,'djokovic wins',igwin,iglose
 else
 if (info ==1)write(*,*) 'end of match',i1,'djokovic lose',igwin,iglose
endif


!write(*,*)'final score',igwin,iglose
enddo !end match


probscore = 1.*idcorscore/nmatch

if (info_out ==1) then
open(unit=1,file='mc_tennis_summary.txt')
do it = 1,nmatch
write(1,*) igwin_save(it),iglos_save(it)
enddo
close(1)
endif

write(*,*) 'matches played...',nmatch
write(*,*) 'Djokovic won...',idjwin
write(*,*) 'djokovic lose..',nmatch-idjwin
write(*,*) 'win frac...',1.*idjwin/nmatch



end subroutine