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
    void pre_exchange();
    void initial_integrate(int);
    void post_force(int);
    virtual void end_of_step();
    int pack_forward_comm(int, int *, double *, int, int *);
    void unpack_forward_comm(int, int, double *);
    double memory_usage();
    void reset_accumulators();
    
  private:
    void findvirtualsite(int, tagint, bool,int);
    void computemoldesum();
    void addforcetomonomers_virtualsitesarenotparticles();
    void updatevirtualparticletype();
    void normalize(double *);
    void computecross(double *, double *, double *);
    double computedot(double *,double *);
    double computedihedral(double *, double *, double *, double *);
    void rotatepointalongaxisbyangle(double *, double *, double, double *);
    void set_filamentID(int,long int,double);
    void findcoordsforpolym(long int);
    int getnewatomtype(int,int,int,int,int);
    int allowtransition(int,int,int);
    int me, nprocs;
    int nlevels_respa;
    int nspecies;
    double ***xvirt;
    int *nspecies_beads;
    int *nstates;
    int nstates_total;
    int *atomtype_offset;
    int ntypereal; //stores the summation of types of atoms in real molecules
    int *bondtype_offset; 
    bool allocated;
    double **rates; //transition rates between states. if rates differ along filaments, enter the value corres. to barbed end.
    double **mhcorr; //Metropolis-Hasting correction term between states
    int **reax_dir; //direction of reactions -1 for forward and 1 for backward
    double **scaling;  //to accout for rate differences in barbed and pointed end. Can also be used other components at later stages
    unsigned short int detailedbalanceflag;//Eqn 11 in JCTC, 2014 DOI: 10.1021/ct500834t
    unsigned short int polymerizationflag; // 1 if polymerization on or 0
    double **eta; //width of the switching function
    double **phi0; //dihedral dependence parameter phi0, where k[][] = rates[][](0.5+0.5*tanh(phi-phi0)), phi being the instantaneous dihedral.
    unsigned short int restrictmolflag;//restrict transitions to certain mols
    int *restrictmol;//list of molecules allowed to transition
    class RanMars *random; //random number generator class

//////
    class RanPark *randomp;
//////


    int tcomputeflag;//temperature compute
    char *id_temp;//temperature compute
    class Compute *temperature; //temperature->compute_scalar()
    double t_current;//temperature at current step, computed at end_of_step

    double **cutsq; //cutoff
    class NeighList *list; //neighbor list class
    int nmol; //stores the total number of molecules based on the initial datafile (includes virtual sites if polymerization enabled)
    int nmolreal; //stores the actual number of real molecules. all non-negative states in initialstate.in file
    //unsigned short int *trans_flag, *trans_flag_global;
    
    short int *trans_flag;
    short int *mol_state;
    unsigned short int *mol_species; //stores the species ID of each molecule
    unsigned long int *mol_endid;
    int *mol_filamentID; //stores the filament ID from 1 to N for actin subunits based on initialstate.in) Other species including g-actin are assigned negative numbers.
    int *mol_filamentID_poly; 
    int *mol_filamentID_depoly;
    int *mol_filamentlistforward;
    int *mol_filamentlistbackward;
    //int *atom_filamentID; //stores the filament ID of each atom in the subunit. atom level info of mol_filamentID.  
    int nfilaments;
    int **filament_ends; //stores the mol ID of barbed end (colIndex 0) and pointed end (colIndex 1) for actin filaments;
    int **filament_endspolymflag;//stores 1 or 0 depending on if a monomer is idenitifed to be close to barbed or pointed end, also prevents two monomers from polymerizing at the same end
    int **filament_endsdepolymflag;
    double *mol_desum,*mol_desum_global;
    unsigned short int *mol_accept;
    double ***xdihed; //coordinates for dihedral
    double **xpolym; //coordinates of actin to check for polymerization
    unsigned short int *xpolymtype; //atom type of actin middle atom ID
    double *mol_dihed; //dihedral angle
    double *atom_state; //per-atom vector to output state of atom, i.e. working copy of vector_atom for the fix
    int migrateatomsflag;
    int *timer; //stores the timestep value at which a monomer depolymerizes
    double *interaction_factor; //stores 0 or 1 to turnoff/on pair wise interactions with virtual sites after depolymerization
    double Beta;

    inline int sbmask(int j) {
      return j >> SBBITS & 3;
    }//similar to pair.h
  protected:
    int nmax;
  };
}
#endif
#endif
