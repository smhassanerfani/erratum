program sum

    implicit none

    ! Variable Declaration
    real :: x, y, ans

    print *, 'Enter two number: '
    
    read *, x
    read *, y

    ans = x + y

    print *, 'The total value is: ', ans

end program sum