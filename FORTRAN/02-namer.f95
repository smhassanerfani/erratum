PROGRAM NameR
    ! This checks if the variables are defined correctly.
    IMPLICIT NONE

    ! Variables Declaration
    CHARACTER :: name * 10

    ! Ask for writing the name
    PRINT *, 'What is your name?'

    ! Read the name
    READ *, name

    ! Print the name
    PRINT *, 'My name is ', name

END PROGRAM NameR
