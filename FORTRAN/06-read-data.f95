program readdata

    implicit none

    ! Variables Declaration
    real :: x, y, z, k

    open(10, file='data.txt')
    read(10, *) x, y, z, k

    print *, x, y, z, k

end program readdata