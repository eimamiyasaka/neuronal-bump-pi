!----------------------------------------------------------------------
!   bump.f90  —  AUTO-07p model: gamma (oscillon) Hopf of the next-gen
!   theta-neuron field bump along the shunting axis g0.
!
!   This is a DIRECT port of the reduced even-grid RHS validated against the
!   project's verified Julia field code to 1.3e-15 (scripts/auto_export_bump.jl).
!   State: complex order parameter z_j = u_j + i w_j on the reflection-symmetric
!   half grid x_j = 2*pi*(j-1)/N, j=1..H, x in [0,pi], N = 2*(H-1), H = NDIM/2.
!   Even symmetry (z_j = z_{N+2-j}) removes the translation Goldstone mode, so the
!   only oscillatory instability detected is the even breathing mode = the oscillon.
!
!   Matched reversals E_E,E_I = (kappa/g0, -kappa/g0): g0 cancels from the drive and
!   survives only in the shunt, so PAR(1)=g0 multiplies a single linear term.
!     M  = 2*pi/N
!     P_j = 1 - (4/3)u_j + (1/3)(u_j^2 - w_j^2)
!     S0 = P_1 + P_H + 2*sum_{2..H-1} P_j
!     Sc = P_1 - P_H + 2*sum_{2..H-1} P_j cos x_j        (cos x_1=1, cos x_H=-1; Ss=0)
!     f_j = etabar + kappa*M*(0.1*S0 + 0.3*cos x_j*Sc)   (g0-independent drive)
!     G_j = g0   *M*(0.7*S0 + 0.3*cos x_j*Sc)            (shunt, linear in g0)
!     zdot_j = (i/2)(z_j-1)^2 - (1/2)(z_j+1)^2 (Delta + i f_j) + (G_j/2)(z_j^2 - 1)
!
!   PAR(1)=g0 (primary), PAR(2)=Delta (2nd, for the (g0,Delta) Hopf locus),
!   PAR(3)=etabar, PAR(4)=kappa.
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      INTEGER H, N, J
      DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979323846D0
      DOUBLE PRECISION G0, DELTA, ETABAR, KAPPA, M
      DOUBLE PRECISION S0, SC, PJ, CJ, UJ, WJ, FJ, GJ
      DOUBLE COMPLEX Z, ZD, IUNIT

      IUNIT  = (0.0D0, 1.0D0)
      G0     = PAR(1)
      DELTA  = PAR(2)
      ETABAR = PAR(3)
      KAPPA  = PAR(4)

      H = NDIM/2
      N = 2*(H-1)
      M = 2.0D0*PI/DBLE(N)

!     --- rank-3 global sums over the full ring via even symmetry ---
      S0 = 0.0D0
      SC = 0.0D0
      DO J = 1, H
         UJ = U(2*J-1)
         WJ = U(2*J)
         PJ = 1.0D0 - (4.0D0/3.0D0)*UJ + (1.0D0/3.0D0)*(UJ*UJ - WJ*WJ)
         CJ = COS(2.0D0*PI*DBLE(J-1)/DBLE(N))
         IF (J.EQ.1 .OR. J.EQ.H) THEN
            S0 = S0 + PJ
            SC = SC + PJ*CJ
         ELSE
            S0 = S0 + 2.0D0*PJ
            SC = SC + 2.0D0*PJ*CJ
         END IF
      END DO

!     --- local dynamics ---
      DO J = 1, H
         UJ = U(2*J-1)
         WJ = U(2*J)
         CJ = COS(2.0D0*PI*DBLE(J-1)/DBLE(N))
         FJ = ETABAR + KAPPA*M*(0.1D0*S0 + 0.3D0*CJ*SC)
         GJ = G0     *M*(0.7D0*S0 + 0.3D0*CJ*SC)
         Z  = DCMPLX(UJ, WJ)
         ZD = 0.5D0*IUNIT*(Z-1.0D0)**2                                   &
     &        - 0.5D0*(Z+1.0D0)**2*DCMPLX(DELTA, FJ)                     &
     &        + 0.5D0*GJ*(Z*Z - 1.0D0)
         F(2*J-1) = DREAL(ZD)
         F(2*J)   = DIMAG(ZD)
      END DO

      END SUBROUTINE FUNC
!----------------------------------------------------------------------
      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     Starting solution: the validated static bump on the half grid, read from
!     auto/bump_init.dat (written by scripts/auto_export_bump.jl). AUTO is invoked
!     from the auto/ directory so the relative path resolves.
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      INTEGER H, NX, J, IOS
      DOUBLE PRECISION ETABAR, DELTA, KAPPA, G0, UU, WW
      CHARACTER(LEN=256) LINE

      OPEN(UNIT=12, FILE='bump_init.dat', STATUS='OLD', ACTION='READ')
      READ(12,'(A)') LINE                         ! comment
      READ(12,'(A)') LINE                         ! comment
      READ(12,*) H, NX, ETABAR, DELTA, KAPPA, G0  ! param line
      DO J = 1, H
         READ(12,*) UU, WW
         U(2*J-1) = UU
         U(2*J)   = WW
      END DO
      CLOSE(12)

      PAR(1) = G0
      PAR(2) = DELTA
      PAR(3) = ETABAR
      PAR(4) = KAPPA

      END SUBROUTINE STPNT
!----------------------------------------------------------------------
      SUBROUTINE PVLS(NDIM,U,PAR)
!     Output extra observables along the branch: peak firing rate and bump
!     contrast, so the 1-parameter diagram can be plotted directly.
!       rate(z) = (1/pi) Re[(1-z)/(1+z)]
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
      INTEGER H, J
      DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979323846D0
      DOUBLE PRECISION RJ, RMAX, RMIN
      DOUBLE COMPLEX Z
      H = NDIM/2
      RMAX = -1.0D30
      RMIN =  1.0D30
      DO J = 1, H
         Z  = DCMPLX(U(2*J-1), U(2*J))
         RJ = (1.0D0/PI)*DREAL((1.0D0-Z)/(1.0D0+Z))
         IF (RJ.GT.RMAX) RMAX = RJ
         IF (RJ.LT.RMIN) RMIN = RJ
      END DO
      PAR(11) = RMAX          ! peak rate
      PAR(12) = RMAX - RMIN   ! bump contrast
      END SUBROUTINE PVLS
!----------------------------------------------------------------------
      SUBROUTINE BCND(NDIM,PAR,ICP,NBC,U0,U1,FB,IJAC,DBC)
      INTEGER, INTENT(IN) :: NDIM, ICP(*), NBC, IJAC
      DOUBLE PRECISION, INTENT(IN) :: PAR(*), U0(NDIM), U1(NDIM)
      DOUBLE PRECISION, INTENT(OUT) :: FB(NBC)
      DOUBLE PRECISION, INTENT(INOUT) :: DBC(NBC,*)
      END SUBROUTINE BCND

      SUBROUTINE ICND(NDIM,PAR,ICP,NINT,U,UOLD,UDOT,UPOLD,FI,IJAC,DINT)
      INTEGER, INTENT(IN) :: NDIM, ICP(*), NINT, IJAC
      DOUBLE PRECISION, INTENT(IN) :: PAR(*),U(NDIM),UOLD(NDIM)
      DOUBLE PRECISION, INTENT(IN) :: UDOT(NDIM),UPOLD(NDIM)
      DOUBLE PRECISION, INTENT(OUT) :: FI(NINT)
      DOUBLE PRECISION, INTENT(INOUT) :: DINT(NINT,*)
      END SUBROUTINE ICND

      SUBROUTINE FOPT(NDIM,U,ICP,PAR,IJAC,FS,DFDU,DFDP)
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: FS
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM), DFDP(*)
      END SUBROUTINE FOPT
!----------------------------------------------------------------------
