program table

    implicit none

    ! Variables Declaration
    real :: x, y, z

    open(10, file='table.txt')
    write(10, *) 'x          y          z'

    do x=1, 5
        do y=1, 10, 0.5
            z = x * y
            print *, x, y, z
            write(10, *)x, y, z
        end do
    end do

end program table