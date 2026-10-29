/* Preload shim: the Ubuntu xfoil build is compiled with -ffpe-trap, so the
   viscous solver dies on harmless NaNs/overflows. Skip the trap setup. */
void _gfortran_set_fpe(int val) { (void)val; }
