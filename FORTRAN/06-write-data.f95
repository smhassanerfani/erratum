program writedata

    implicit none

    ! Variables Declaration
    real :: x, y, z

    x = 0.35
    y = 1.25
    z = 123.0

    open(12, file='data-02.txt')
    write(12, *) x, y, z

    print *, 'Variables have been written in data-02.txt'

end program writedata