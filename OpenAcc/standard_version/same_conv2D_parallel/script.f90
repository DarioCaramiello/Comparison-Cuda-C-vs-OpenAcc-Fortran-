program program_conv
   implicit none

   integer, parameter :: N = 5     
   integer, parameter :: Q = 3       
   integer, parameter :: K = 1

   integer :: matrix(N, N)
   integer :: kernel(Q*Q)
   integer, allocatable :: result(:,:)

   integer :: transfer_time_start_copyin, transfer_time_end_copyin, rate_transfer_copyin
   integer :: transfer_time_start_copyout, transfer_time_end_copyout, rate_transfer_copyout
   integer :: start_comp_time, end_comp_time, rate_comp

   real(8) :: elapsed_time_copyin
   real(8) :: elapsed_time_copyout
   real(8) :: elapsed_time_comp

   integer :: i, j, r, c, nx, ny, index, sum

   ! Inizializza la matrice e il kernel
   call initialize_matrix(matrix, N, N)
   !print *, "Matrix : "
   !call show_matrix(matrix, N, N)
   !print *, "Kernel : "
   call generate_sobel_kernel(kernel, Q)

   allocate(result(N, N))

   call system_clock(transfer_time_start_copyin, rate_transfer_copyin)
   !$acc data copyin(matrix, kernel) copyout(result)
   !$acc wait 
   call system_clock(transfer_time_end_copyin, rate_transfer_copyin)


   call system_clock(start_comp_time, rate_comp)
   !$acc parallel loop collapse(2) private(i,j,r,c,nx,ny,index,sum) present(matrix, kernel, result)

   do i = 1, N
      do j = 1, N

         sum = 0

         do r = -K, K
            do c = -K, K

               nx = i + r
               ny = j + c

               if ( nx >= 1 .and. nx <= N .and. ny >= 1 .and. ny <= N ) then
                  index = (r + K)*Q + (c + K) + 1
                  sum = sum + matrix(nx, ny) * kernel(index)
               end if

            end do
         end do
            
         result(i,j) = -sum 

      end do
   end do
   !$acc end parallel loop

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

    subroutine initialize_matrix(mat, M, N)
        integer, intent(out) :: mat(M, N)
        integer, intent(in)  :: M, N
        integer :: i, j
        do i = 1, M
           do j = 1, N
              mat(i,j) = i + j - 1
           end do
        end do
    end subroutine initialize_matrix

    subroutine generate_sobel_kernel(k, Q)
        integer, intent(out) :: k(Q*Q)
        integer, intent(in)  :: Q
      
         k(1) = 1
         k(2) = 0
         k(3) = -1
         k(4) = 2
         k(5) = 0
         k(6) = -2
         k(7) = 1
         k(8) = 0
         k(9) = -1
        
    end subroutine generate_sobel_kernel

    !-----------------------------------------------------------
    ! Stampa una matrice intera
    subroutine show_matrix(mat, M, N)
        integer, intent(in) :: mat(M, N)
        integer, intent(in) :: M, N
        integer :: i, j
        do i = 1, M
           do j = 1, N
              write(*,'(I5)', advance="no") mat(i,j)
           end do
           print *
        end do
        print *
    end subroutine show_matrix

end program program_conv
