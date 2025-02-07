

program program_conv 

    implicit none

    real(8) :: start_time, end_time
    
    integer, parameter :: N = 5
    integer, parameter :: Q = 3
    integer, parameter :: K = 1

    integer :: matrix(N, N)
    integer :: kernel(Q*Q)
    integer, allocatable :: result(:, :)

    integer :: transfer_time_start_copyin, transfer_time_end_copyin, rate_transfer_copyin
    integer :: transfer_time_start_copyout, transfer_time_end_copyout, rate_transfer_copyout
    integer :: start_comp_time, end_comp_time, rate_comp

    real(8) :: elapsed_time_copyin
    real(8) :: elapsed_time_copyout
    real(8) :: elapsed_time_comp

    integer :: vec(9)
    integer :: sum, x, i, j, r, c
    integer :: nx, ny
    integer :: index_out, size_vec_neighborhood
    size_vec_neighborhood = ((2*K+1) * (2*K+1))


    call initialize_matrix(matrix, N, N)
    !print *, "Matrix : "
    !call show_matrix(matrix, N, N)

    call generate_sobel_kernel(kernel, Q)
    !print *, "Kernel : "
    !call show_matrix(kernel, Q, Q)

    !call system_clock(start_comp, rate_comp)
    !call same_conv_2D(matrix, N, N, kernel, Q, Q, K, result)
    !call system_clock(end_comp, rate_comp)

    allocate(result(N, N))
        
    index_out = 1

    call system_clock(transfer_time_start_copyin, rate_transfer_copyin)
    !$acc data copyin(matrix, kernel) copyout(result)

    !$acc wait 
    call system_clock(transfer_time_end_copyin, rate_transfer_copyin)

    
    call system_clock(start_comp_time, rate_comp)
    !$acc parallel 
    
    !$acc loop collapse(2) private(vec)
    do i=1, N 
        do j=1, N
            do r=-K, K
                do c=-K, K
                    
                    nx = i+r
                    ny = j+c

                    if( nx>=1 .and. nx<=N .and. ny>=1 .and. ny<=N) then
                        vec(index_out) = matrix(nx, ny)
                    else
                        vec(index_out) = 0
                    end if
                    index_out = index_out + 1
                end do 
            end do  

            sum = 0
            do x=1, size_vec_neighborhood
                sum = sum + (vec(x) * kernel(x))
            end do      
            result(i, j) = -sum

            vec = 0
            index_out = 1

        end do
    end do

    !$acc end parallel
    
    !$acc wait

    call system_clock(end_comp_time, rate_comp)



    call system_clock(transfer_time_start_copyout, rate_transfer_copyout)
    !$acc end data

    !$acc wait 
    
    call system_clock(transfer_time_end_copyout, rate_transfer_copyout)


    print *, "Result : "
    call show_matrix(result, N, N)


    elapsed_time_copyin = ( real(transfer_time_end_copyin - transfer_time_start_copyin) / real(rate_transfer_copyin))
    elapsed_time_copyout = ( real(transfer_time_end_copyout - transfer_time_start_copyout) / real(rate_transfer_copyout))
    elapsed_time_comp = ( real(end_comp_time - start_comp_time) / real(rate_comp))
    !elapsed_time_program = ( real(end_program - start_program) / real(rate_program) )
    !elapsed_time_comp = ( real(end_comp - start_comp) / real(rate_comp) )

    !print *, "Execution copyin time : "
    !print '(F12.8)', elapsed_time_copyin
    !print *, "Execution copyout time : "
    !print '(F12.8)', elapsed_time_copyout

    print *, "Execution parallel comp time : "
    print '(F12.8)', elapsed_time_comp

    print *, "Execution memory time (host --> device)(device --> host) : "
    print '(F12.8)', elapsed_time_copyin + elapsed_time_copyout

    print *, "Execution tot  time : "
    print '(F12.8)', elapsed_time_comp + elapsed_time_copyin + elapsed_time_copyout




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

        !kernel(1) = 1 
        !kernel(2) = 2
        !kernel(3) = 1
        !kernel(4) = 0
        !kernel(5) = 0
        !kernel(6) = 0
        !kernel(7) = -1
        !kernel(8) = -2
        !kernel(9) = -1


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


        !$acc data copyin(matrix, kernel) copyout(result)
        
        !$acc parallel 
        
        !$acc loop collapse(2) private(vec)
        do i=1, rows 
            do j=1, cols
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

                sum = 0
                do k=1, size_vec_neighborhood
                    sum = sum + (vec(k) * kernel(k))
                end do      
                result(i, j) = -sum

                vec = 0
                index_out = 1

            end do
        end do

        !$acc end parallel

        !$acc end data


    end subroutine same_conv_2D

end program program_conv