!----------------------------------------------------------------------
!   homog.f90  —  AUTO-07p model: the SPATIALLY-UNIFORM conductance mean field.
!
!   Warm-up / baseline. This is the g0-reduced field of bump.f90 collapsed to a
!   single uniform population (Sc = 0, S0 = N*P), i.e. NDIM = 2 (z = u + i w):
!     P   = 1 - (4/3)u + (1/3)(u^2 - w^2)
!     f   = etabar + 0.2*pi*kappa*P        (matched-reversal drive, g0-independent)
!     G   = 1.4*pi*g0*P                    (uniform shunt, linear in g0)
!     zdot = (i/2)(z-1)^2 - (1/2)(z+1)^2 (Delta + i f) + (G/2)(z^2 - 1)
!
!   Purpose: (a) verify the AUTO toolchain end-to-end on the project's own equations
!   on a trivially small system; (b) map the homogeneous (bulk) bifurcations — the
!   saddle-node bistability and any bulk Hopf — which are DISTINCT from the localized
!   oscillon Hopf of bump.f90 (CLAUDE.md: the sustained gamma rides the bump, not the
!   uniform state). PAR(1)=g0, PAR(2)=Delta, PAR(3)=etabar, PAR(4)=kappa.
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979323846D0
      DOUBLE PRECISION G0, DELTA, ETABAR, KAPPA, UU, WW, P, FF, GG
      DOUBLE COMPLEX Z, ZD, IUNIT

      IUNIT  = (0.0D0, 1.0D0)
      G0     = PAR(1)
      DELTA  = PAR(2)
      ETABAR = PAR(3)
      KAPPA  = PAR(4)

      UU = U(1)
      WW = U(2)
      P  = 1.0D0 - (4.0D0/3.0D0)*UU + (1.0D0/3.0D0)*(UU*UU - WW*WW)
      FF = ETABAR + 0.2D0*PI*KAPPA*P
      GG = 1.4D0*PI*G0*P

      Z  = DCMPLX(UU, WW)
      ZD = 0.5D0*IUNIT*(Z-1.0D0)**2                                     &
     &     - 0.5D0*(Z+1.0D0)**2*DCMPLX(DELTA, FF)                       &
     &     + 0.5D0*GG*(Z*Z - 1.0D0)
      F(1) = DREAL(ZD)
      F(2) = DIMAG(ZD)

      END SUBROUTINE FUNC
!----------------------------------------------------------------------
      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     Start from the coupled uniform equilibrium computed in Julia
!     (scripts/auto_export_homog.jl -> homog_init.dat). AUTO runs from auto/.
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T
      DOUBLE PRECISION ETABAR, DELTA, KAPPA, G0, UU, WW
      CHARACTER(LEN=256) LINE
      OPEN(UNIT=13, FILE='homog_init.dat', STATUS='OLD', ACTION='READ')
      READ(13,'(A)') LINE
      READ(13,'(A)') LINE
      READ(13,*) ETABAR, DELTA, KAPPA, G0
      READ(13,*) UU, WW
      CLOSE(13)
      PAR(1) = G0
      PAR(2) = DELTA
      PAR(3) = ETABAR
      PAR(4) = KAPPA
      U(1) = UU
      U(2) = WW
      END SUBROUTINE STPNT
!----------------------------------------------------------------------
      SUBROUTINE PVLS(NDIM,U,PAR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)
      DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979323846D0
      DOUBLE COMPLEX Z
      Z = DCMPLX(U(1), U(2))
      PAR(11) = (1.0D0/PI)*DREAL((1.0D0-Z)/(1.0D0+Z))   ! uniform firing rate
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
