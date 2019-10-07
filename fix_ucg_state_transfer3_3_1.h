#ifdef FIX_CLASS

FixStyle(ucg_state_trans3_3_1,FixUCGStateTrans3_3_1)

#else

#ifndef LMP_FIX_UCG_STATE_TRANS3_3_1_H
#define LMP_FIX_UCG_STATE_TRANS3_3_1_H

#include "fix.h"

namespace LAMMPS_NS {
  class FixUCGStateTrans3_3_1 : public Fix {
  public:
    FixUCGStateTrans3_3_1(class LAMMPS *, int, char **);
    ~FixUCGStateTrans3_3_1();
    int setmask();
    void consistencycheck();
    void allocate();
    void init();
    void init_list(int, class NeighList *);
    void setup(int);
    void post_force_respa(int, int, int);
    void post_force(int);
    virtual void end_of_step();
    int pack_forward_comm(int, int *, double *, int, int *);
    void unpack_forward_comm(int, int, double *);
    double memory_usage();
    void reset_accumulators();
    
  private:

    // user-defined functions
    //-------------------------
    void normalize(double *);
    void computecross(double *, double *, double *);
    double computedot(double *,double *);
    double computedihedral(double *, double *, double *, double *);
    void rotatepointalongaxisbyangle(double *, double *, double, double *);
    void computemoldesum();
    int getnewatomtype(int,int,int);

    // variables associated with reading inputs
    //-----------------------------------------
    int nspecies;
    int nstates_total;
    int ntypereal;    //stores the summation of types of atoms in real molecules 
    bool allocated;
    unsigned short int restrictmolflag;//restrict transitions to certain mols
    int nmol;    //stores the total number of molecules based on the initial datafile (includes virtual sites if polymerization enabled)                                                                                                                          
    int nmolreal;    //stores the actual number of real molecules. all non-negative states in initialstate.in file
    int migrateatomsflag;
    double Beta;

    // variable allocated in the memory->create function
    //-------------------------------------------------
    int *nspecies_beads;
    unsigned long int *mol_endid;
    unsigned short int *mol_species;    //stores the species ID of each molecule
    int *atomtype_offset;
    int *bondtype_offset;
    int *nstates;
    short int *trans_flag;
    short int *mol_state;
    double *mol_desum;
    unsigned short int *mol_accept;
    int **reax_dir;    //direction of reactions -1 for forward and 1 for backward 
    double **rates;    //the prefactor k0  between states
    double **mhcorr;    //Metropolis-Hasting correction term between states 
    double *mol_desum_global;
    int *restrictmol;    //list of molecules allowed to transition

    // LAMMPS related and other variables
    //----------------------------------
    class RanMars *random;    //random number generator class
    class RanPark *randomp;
    int me, nprocs;
    int nlevels_respa;
    int tcomputeflag;    //temperature compute
    char *id_temp;    //temperature compute
    class Compute *temperature;    //temperature->compute_scalar()
    double t_current;    //temperature at current step, computed at end_of_step
    double **cutsq;    //cutoff
    class NeighList *list;    //neighbor list class

    inline int sbmask(int j) {
      return j >> SBBITS & 3;
    }//similar to pair.h
  protected:
    int nmax;
  };
}
#endif
#endif
