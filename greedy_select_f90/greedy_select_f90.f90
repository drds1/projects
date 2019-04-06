subroutine getcisq(true,predict,N,cisq)
    integer,intent(in)::N
    doubleprecision,intent(in):: true(N),predict(N)
    doubleprecision,intent(out):: cisq
    doubleprecision:: res
    !f2py intent(in):: true,predict
    !f2py intent(hide), depend(true):: n = shape(true)
    !f2py intent(out) cisq
    cisq=0.d0
    do i=1,N
        res = true(i) - predict(i)
        cisq = cisq + res*res
    end do
end subroutine getcisq








subroutine inverse(a,c,n)
!============================================================
! Inverse matrix
! Method: Based on Doolittle LU factorization for Ax=b
! Alex G. December 2009
!-----------------------------------------------------------
! input ...
! a(n,n) - array of coefficients for matrix A
! n      - dimension
! output ...
! c(n,n) - inverse matrix of A
! comments ...
! the original matrix a(n,n) will be destroyed
! during the calculation
!===========================================================
implicit none
integer n
double precision,dimension(n,n):: a, c
double precision L(n,n), U(n,n), b(n), d(n), x(n)
double precision coeff
integer i, j, k

!f2py intent(in):: a
!f2py intent(hide), depend(a):: n = shape(a,0)
!f2py intent(out) c

! step 0: initialization for matrices L and U and b
! Fortran 90/95 aloows such operations on matrices
L=0.0
U=0.0
b=0.0

! step 1: forward elimination
do k=1, n-1
   do i=k+1,n
      coeff=a(i,k)/a(k,k)
      L(i,k) = coeff
      do j=k+1,n
         a(i,j) = a(i,j)-coeff*a(k,j)
      end do
   end do
end do

! Step 2: prepare L and U matrices
! L matrix is a matrix of the elimination coefficient
! + the diagonal elements are 1.0
do i=1,n
  L(i,i) = 1.0
end do
! U matrix is the upper triangular part of A
do j=1,n
  do i=1,j
    U(i,j) = a(i,j)
  end do
end do

! Step 3: compute columns of the inverse matrix C
do k=1,n
  b(k)=1.0
  d(1) = b(1)
! Step 3a: Solve Ld=b using the forward substitution
  do i=2,n
    d(i)=b(i)
    do j=1,i-1
      d(i) = d(i) - L(i,j)*d(j)
    end do
  end do
! Step 3b: Solve Ux=d using the back substitution
  x(n)=d(n)/U(n,n)
  do i = n-1,1,-1
    x(i) = d(i)
    do j=n,i+1,-1
      x(i)=x(i)-U(i,j)*x(j)
    end do
    x(i) = x(i)/u(i,i)
  end do
! Step 3c: fill the solutions x(n) into column k of C
  do i=1,n
    c(i,k) = x(i)
  end do
  b(k)=0.0
end do
end subroutine inverse






subroutine glm_fit(y,X,N,k,parm)
    integer,intent(in)::N,k
    doubleprecision,intent(in)::y(N),X(N,k)
    doubleprecision,intent(out):: parm(k)
    doubleprecision hes(k,k),c(k),cov(k,k)
    doubleprecision sum1,sum2

    call cpu_time(starttime)
    do k1 = 1,k
        !evaluate the c vector
        sum1 = 0.d0
        do i = 1,N
            sum1 = sum1 + y(i)*X(i,k1)
        end do
        c(k1) = sum1

        !evaluate the hessian matrix
        do k2  = 1,k
            sum2 = 0.d0
            do i = 1,N
                sum2 = sum2 + X(i,k2)*X(i,k1)
            end do
            hes(k2,k1) = sum2
        end do
    end do
    call cpu_time(endtime)
    write(*,*) 'make hess',endtime-starttime

    !evaluate the covariance matrix
    call cpu_time(starttime)
    call inverse(hes,cov,k)
    call cpu_time(endtime)
    write(*,*) 'inverting hess',endtime-starttime

    !evaluate the parameters
    do k1 = 1,k
        sum1 = 0.d0
        do k2 = 1,k
            sum1 = sum1 + cov(k2,k1)*c(k2)
        end do
        parm(k1) = sum1
    end do

end subroutine glm_fit





subroutine glm_predict(X,parms,predicted,N,k)
    integer,intent(in)::N,k
    doubleprecision,intent(in):: X(N,k),parms(k)
    doubleprecision,intent(out):: predicted(N)
    doubleprecision:: sum

    do i = 1,N
        sum = 0.d0
        do ik=1,k
            sum = sum + parms(ik)*X(i,ik)
        end do
        predicted(i) = sum
    end do

end subroutine glm_predict



!program test_inverse
!!====================================================================
!!  Computing Inverse matrix
!!  Method: Based on the Doolittle LU method
!!====================================================================
!implicit none
!integer, parameter :: n=3
!real:: starttime,endtime
!double precision a(n,n), c(n,n)
!integer i,j
!! matrix A
!  data (a(1,i), i=1,3) /  3.0,  2.0,  4.0 /
!  data (a(2,i), i=1,3) /  2.0, -3.0,  1.0 /
!  data (a(3,i), i=1,3) /  1.0,  1.0,  2.0 /
!
!! print a header and the original matrix
!  write (*,200)
!  do i=1,n
!     write (*,201) (a(i,j),j=1,n)
!  end do
!
!  call cpu_time(starttime)
!  call inverse(a,c,n)
!  call cpu_time(endtime)
!
!! print the inverse matrix C = A^{-1}
!  write (*,202)
!  do i = 1,n
!     write (*,201)  (c(i,j),j=1,n)
!  end do
!200 format (' Computing Inverse matrix ',/,/, &
!            ' Matrix A')
!201 format (6f12.6)
!202 format (/,' Inverse matrix A^{-1}')
!write(*,*) 'compute time',endtime-starttime
!end

!program testcisq
!
!    real true(10),predict(10)
!    doubleprecision csq
!
!    do i = 1,10
!        true(i) = 1.*i
!    end do
!    predict(1:10) = 5.0
!
!    call getcisq(true,predict,10,csq)
!    write(*,*) csq
!
!end program testcisq


!gfortran greedy_select_f90.f90 f90random.f90
program testglm
    doubleprecision:: cisq
    doubleprecision,allocatable:: X(:,:),Y(:),parms(:),predicted(:),ptrue(:)
    integer N,k,iseed
    N = 2000
    k = 100
    ktrue = 2
    iseed = 324342



    allocate(X(N,k),Y(N),parms(k),predicted(N),ptrue(ktrue))
    ptrue(1) = 20.0
    ptrue(2) = 13.0

!generate fake data input random noise
    do i = 1,N
        Y(i) = 3.2*sin(2*3.1415926535/P1 * i) + 8.0*sin(2*3.1415926535/P2 * i)
        do kx=1,k
            X(i,kx) = c4_normal_01 ( iseed )
        end do
    end do

    !input true signals
    do i = 1,N
        Y(i) = 3.2*sin(2*3.1415926535/ptrue(1) *i) + 8.0*sin(2*3.1415926535/ptrue(2) * i)
        do kx=1,ktrue
            X(i,kx) = sin(2*3.1415926535/ptrue(kx) * i)
        end do
    end do

!run and time fit
    call cpu_time(starttime)
    call glm_fit(y,X,N,k,parms)
    call cpu_time(endtime)
    ttime = endtime-starttime
    write(*,*) 'training time',ttime
    call cpu_time(starttime)
    call glm_predict(X,parms,predicted,N,k)
    call cpu_time(endtime)
    ptime = endtime-starttime
    write(*,*) 'predict time',ptime
    write(*,*)'ratio train/predict',ttime/ptime
    call getcisq(y,predicted,N,cisq)
    write(*,*) 'true, predicted cisq',cisq






end program testglm