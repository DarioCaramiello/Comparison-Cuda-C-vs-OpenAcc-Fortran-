

program program_conv 

    implicit none

    real(8) :: start_time, end_time
    
    integer, parameter :: N = 10000
    integer, parameter :: Q = 3
    integer, parameter :: K = 1

    integer :: matrix(N, N)
    integer :: kernel(Q*Q)
    integer, allocatable :: result(:, :)

    integer :: start_program, end_program, rate_program
    integer :: start_comp, end_comp, rate_comp

    real(8) :: elapsed_time_program
    real(8) :: elapsed_time_comp

    call system_clock(start_program, rate_program)

    call initialize_matrix(matrix, N, N)
    !print *, "Matrix : "
    !call show_matrix(matrix, N, N)

    call generate_sobel_kernel(kernel, Q)
    !print *, "Kernel : "
    !call show_matrix(kernel, Q, Q)

    call system_clock(start_comp, rate_comp)
    call same_conv_2D(matrix, N, N, kernel, Q, Q, K, result)
    call system_clock(end_comp, rate_comp)

    !print *, "Result : "
    !call show_matrix(result, N, N)

    call system_clock(end_program, rate_program)

    elapsed_time_program = ( real(end_program - start_program) / real(rate_program) )
    elapsed_time_comp = ( real(end_comp - start_comp) / real(rate_comp) )

    print *, "Execution time computation : "
    print '(F12.8)', elapsed_time_comp
    print *, "Execution time program : "
    print '(F12.8)', elapsed_time_program



contains

    subroutine initialize_matrix(matrix, M, N)
        integer, intent(out) :: matrix(M, N)
        integer, intent(in) :: M
        integer, intent(in) :: N
        integer :: i,j 

        do i=1, M
            do j=1, N
                matrix(i,j) = i-1 + j-1 + 1
            end do
        end do 
    end subroutine initialize_matrix

    subroutine show_matrix(matrix, M, N)
        integer, intent(out) :: matrix(N, N)
        integer, intent(in) :: M
        integer, intent(in) :: N
        integer :: i,j

        do i=1, M
            do j=1, N
                write(*, "(I5)", advance="no") matrix(i, j)
            end do
            print *
        end do
        print * 
    end subroutine show_matrix 

    subroutine generate_sobel_kernel(kernel, Q)
        integer, intent(in) :: Q
        integer, intent(out) :: kernel(Q*Q)
        integer :: i,j

        kernel(1) = 1
        kernel(2) = 0
        kernel(3) = -1
        kernel(4) = 2
        kernel(5) = 0
        kernel(6) = -2
        kernel(7) = 1
        kernel(8) = 0
        kernel(9) = -1


    end subroutine generate_sobel_kernel

    subroutine same_conv_2D(matrix, rows, cols, kernel, k_rows, k_cols, radius, result)

        integer, intent(in) :: rows, cols
        integer, intent(in) :: k_rows, k_cols
        integer, intent(in) :: radius
        integer, intent(in) :: matrix(rows, cols)
        integer, intent(in) :: kernel(k_rows * k_cols)
        integer, allocatable, intent(out) :: result(:, :)

        integer :: vec(9)
        integer :: sum, k, i, j, r, c
        integer :: nx, ny
        integer :: index_out, size_vec_neighborhood
        size_vec_neighborhood = ((2*radius+1) * (2*radius+1))
      
        allocate(result(rows, cols))
        
        index_out = 1

        do i=1, rows 
            do j=1, cols
                !print *, "elem i, j : "
                !print *, i
                !print *, j

                do r=-radius, radius
                    do c=-radius, radius
                        
                        nx = i+r
                        ny = j+c

                        if( nx>=1 .and. nx<=rows .and. ny>=1 .and. ny<=cols) then
                            vec(index_out) = matrix(nx, ny)
                        else
                            vec(index_out) = 0
                        end if
                        index_out = index_out + 1
                    end do 
                end do 

                
                !print *, "Vicini : "
                !do p=1, 9
                !    write(*, "(I5)", advance="no") vec(p)
                !end do 
                !print * 

                !print *, "Kernel :"
                !do p=1, 9
                !    write(*, "(I5)", advance="no") kernel(p)
                !end do 
                !print * 

                sum = 0
                do k=1, size_vec_neighborhood
                    sum = sum + (vec(k) * kernel(k))
                end do      
                result(i, j) = - sum

                vec = 0
                index_out = 1

            end do
        end do
    end subroutine same_conv_2D

end program program_conv