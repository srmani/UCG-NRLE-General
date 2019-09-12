/*---------------------------------------------------------------------- */
#include "mpi.h"
#include "modify.h"
#include "memory.h"
#include "update.h"
#include "respa.h"
#include "force.h" //force->numeric(), etc
#include "random_mars.h" //random number generator
#include "fix_ucg_state_transfer3_3_1.h"
#include "error.h" //error
#include <string.h> //strcmp()
#include "neighbor.h" //neigh->build
#include "neigh_list.h" //class NeighList
#include "neigh_request.h" //neigh->request
#include "atom.h" //per-atom variables
#include "pair.h" //force->pair
#include "bond.h" //force->bond
#include "group.h"//temperature compute
#include "compute.h" //temperature->compute_scalar
#include "domain.h" //temperature->compute_scalar
#include "comm.h" //temperature->compute_scalar
#include "irregular.h" //addforcetomonomers_virtualsitesareparticles
#include "random_park.h" // substitute for random_mars
#include <fstream> //read input files: rates and mhcorr
#include <string> //read input files: rates and mhcorr
#include <cmath> //read input files: rates and mhcorr

using namespace LAMMPS_NS;
using namespace FixConst; //in fix.h, defines POST_FORCE, etc.

#define MYDEBUG
//#define DIAGNOSTICS
#define PI 3.14159
/* ---------------------------------------------------------------------- */
FixUCGStateTrans3_3_1::FixUCGStateTrans3_3_1(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp,narg,arg)
{
  
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  
  int iarg=0;
  iarg++; // fix-id [0]
  iarg++; // group [1]
  iarg++; // fix-name [2]
  int seed = force->inumeric(FLERR,arg[iarg++]);

  if(strcmp(arg[iarg++],"Beta")!=0) error->all(FLERR,"keyword Beta is missing in fix ucg command");
  Beta = force->numeric(FLERR,arg[iarg++]);

  if(strcmp(arg[iarg++],"nspecies")!=0) error->all(FLERR,"keyword nspecies is missing in fix ucg command");
  nspecies = force->inumeric(FLERR,arg[iarg++]);
  
  if(strcmp(arg[iarg++],"nstates")!=0) error->all(FLERR,"keyword nstates is missing in fix ucg command");
  int *nstates_temp=new int [nspecies];
  nstates_total=0;
  for (int i=0;i<nspecies;i++){
    nstates_temp[i]=force->inumeric(FLERR,arg[iarg++]);
    nstates_total+=nstates_temp[i];
  }

  if (strcmp(arg[iarg++], "speciesbeads") != 0) error->all(FLERR, "keyword speciesbeads is missing in fix ucg command");
  int *nspecies_beads_temp = new int[nspecies];
  for (int i = 0; i<nspecies; i++) {
    nspecies_beads_temp[i] = force->inumeric(FLERR, arg[iarg++]);
  }
  
  if(strcmp(arg[iarg++],"aoffset")!=0) error->all(FLERR,"keyword aoffset (atomtype offset) is missing in fix ucg command");
  int *aoffset_temp=new int[nspecies];
  for(int i=0;i<nspecies;i++){
    aoffset_temp[i]=force->inumeric(FLERR,arg[iarg++]);
  }

  if(strcmp(arg[iarg++],"boffset")!=0) error->all(FLERR,"keyword boffset (bondtype offset) is missing in fix ucg command");
  int *boffset_temp=new int[nspecies];
  for(int i=0;i<nspecies;i++){
    boffset_temp[i]=force->inumeric(FLERR,arg[iarg++]);
  }
 
  {
    //Finding the maximum molecule ID in the structure and storing in variable nmol//
    int nlocal = atom->nlocal;
    tagint *molecule = atom->molecule;	
    int maxmol_local=-1;
    for(int i=0;i<nlocal;i++) {if(molecule[i]>maxmol_local) maxmol_local=molecule[i];}
    MPI_Allreduce(&maxmol_local,&nmol,1,MPI_INT,MPI_MAX,world);
  }
    
  //Allocate arrays
  allocated=false;  allocate();

  for(int i=0;i<nspecies;i++){
    nstates[i]=nstates_temp[i];
  }
  delete [] nstates_temp;

  for (int i = 0; i<nspecies; i++) {
    nspecies_beads[i] = nspecies_beads_temp[i];
  }
  delete[] nspecies_beads_temp;

  for(int i=0;i<nspecies;i++){
    atomtype_offset[i]=aoffset_temp[i];
  }
  delete [] aoffset_temp;

  for(int i=0;i<nspecies;i++){
    bondtype_offset[i]=boffset_temp[i];
  }
  delete [] boffset_temp;

  
  //Read UCG related inputs by processor 0. Bcast required variables
  if(me==0)
    {
      for(int i=0;i<nstates_total;i++){for(int j=0;j<nstates_total;j++) rates[i][j] = mhcorr[i][j] = 0.0;} //default rates and mhcorr

      while(iarg<narg)
        {
	  if(strcmp(arg[iarg],"rates")==0)
            {
	      iarg++;
	      if(narg<iarg+1) error->one(FLERR,"illegal fix ucg command-missing input");
	      {
		double temprate;
		std::ifstream ratesfile(arg[iarg++]);
		if(!ratesfile) {error->one(FLERR,"rate input file missing in the working directory");}
		for(int i=0;i<nstates_total;i++)
		  for(int j=0;j<nstates_total;j++)
		    { 
		      ratesfile>>temprate; //change to check if input is a number 
		      rates[i][j] = temprate;
		      if(rates[i][j]>1.0 || rates[i][j]<0) error->all(FLERR,"illegal input: expected rate input in interval (0 1)");
		    }
		ratesfile.close();
	      }
            }

	  else if(strcmp(arg[iarg],"mhcorr")==0)//
            {
	      iarg++;
	      if(narg<iarg+1) error->one(FLERR,"illegal fix ucg command-missing input");
	      {
		double tempmhcorr;
		std::ifstream mhcorrfile(arg[iarg++]);
		if(!mhcorrfile) {error->one(FLERR,"mhcorr input file missing in the working directory");}
		for(int i=0;i<nstates_total;i++)
		  for(int j=0;j<nstates_total;j++)
		    { 
		      mhcorrfile>>tempmhcorr; //change to check if input is a number
		      mhcorr[i][j]=tempmhcorr;
		    }
		mhcorrfile.close();
	      }
            }
	  
	  else if (strcmp(arg[iarg], "initialstatefile") == 0)
            {
	      iarg++;
	      if (narg<iarg+1)  error->one(FLERR, "illegal fix ucg command-missing input");
	      {
		std::ifstream istatefile(arg[iarg++]);
		if (!istatefile) { error->one(FLERR, "initialstate file missing in the working directory"); }
		std::string istate0;
		char *istate1;
		int cumul_beads = 0;
		nmolreal=0;
		for (int imol=1;imol<(nmol+1);imol++)
		  {
		    istatefile >> istate0;
		    istate1 = (char *)alloca(istate0.size() + 1);
		    memcpy(istate1, istate0.c_str(), istate0.size() + 1);
		    mol_state[imol] = force->inumeric(FLERR, istate1); //change to check if input is a number
		    if(mol_state[imol]>=0) 
		      nmolreal++; 
		    int cumul_states = 0;
		    for (int ispecies = 0; ispecies < nspecies; ispecies++)
		      {
			cumul_states += nstates[ispecies];
			if (mol_state[imol] < cumul_states)
			  {
			    mol_species[imol] = ispecies;
			    break;
			  }
		      }
		    cumul_beads = cumul_beads + nspecies_beads[mol_species[imol]];
		    mol_endid[imol] = cumul_beads;
		  }
		istatefile.close();
	      }

#ifdef DIAGNOSTICS
	      std::ofstream write("molinfo_initial");
	      write<<"molID"<<' '<<"mol_state"<<' '<<"mol_species"<<' '<<"mol_endid"<<"\n";
	      for(int imol=1;imol<(nmol+1);imol++)
		write<<imol<<' '<<mol_state[imol]<<' '<<mol_species[imol]<<' '<<mol_endid[imol]<<"\n";
	      write.close();
#endif
            }

	  else if (strcmp(arg[iarg], "restrictmol") == 0)
            {
	      iarg++;
	      if (narg < iarg + 1) error->one(FLERR, "illegal fix ucg command-missing input");
	      {
		restrictmolflag = 1;
		std::ifstream restrictmolfile(arg[iarg++]);
		std::string temprestrictmol;
		char *temprestrictmol1;
		if (!restrictmolfile) { error->one(FLERR, "restrictmol file missing in the working directory"); }
		for (int imol=1;imol<(nmol+1); imol++)
		  {
		    restrictmolfile >> temprestrictmol;
		    temprestrictmol1 = (char *)alloca(temprestrictmol.size() + 1);
		    memcpy(temprestrictmol1, temprestrictmol.c_str(), temprestrictmol.size() + 1);
		    restrictmol[imol] = force->inumeric(FLERR, temprestrictmol1);
		  }
		restrictmolfile.close();
	      }
            }

	  else if (strcmp(arg[iarg], "direction") == 0) 
            {
	      iarg++;
	      if (narg<iarg+1) error->one(FLERR, "illegal fix ucg command-missing input");
	      {
		int Val;
		std::ifstream dirfile(arg[iarg++]);
		if (!dirfile) { error->one(FLERR, "direction file missing in the working directory"); }
		for (int i = 0; i < nstates_total; i++) {
		  for (int j = 0; j < nstates_total; j++)
		    {
		      dirfile >> Val;
		      reax_dir[i][j] = Val;
		    }
		}
		dirfile.close();
	      }
            }

	  else
	    error->one(FLERR, "illegal fix ucg command-unknown keyword");
        }
    }
	 
  MPI_Bcast(&mol_state[0],nmol+1,MPI_SHORT,0,world);
  MPI_Bcast(&mol_species[0],nmol+1,MPI_UNSIGNED_SHORT,0,world);
  MPI_Bcast(&mol_endid[0], nmol+1, MPI_UNSIGNED_LONG, 0, world);
  MPI_Bcast(&nmolreal,1,MPI_INT,0,world);
 
  tagint *molecule = atom->molecule;
  for(int i=0;i<atom->nlocal;i++) {int imol=molecule[i]; atom_state[i]=mol_state[imol];}

  //check for pair style with cutoff
  if (force->pair == NULL) error->all(FLERR,"fix ucg requires a pair style");
  if (force->pair->cutsq == NULL) error->all(FLERR,"fix ucg is incompatible with pair style");
  cutsq = force->pair->cutsq;
  
  //create a new compute temp style, similar to fix_langevin
  int n = strlen(id) + 6;
  id_temp = new char[n];
  strcpy(id_temp,id);
  strcat(id_temp,"_temp");
  char **newarg = new char*[3];
  newarg[0] = id_temp;
  newarg[1] = group->names[igroup];
  newarg[2] = (char *) "temp";
  modify->add_compute(3,newarg);
  delete [] newarg;
  tcomputeflag = 1;
  
  //initialize the rng
  random = new RanMars(lmp, seed);
  randomp = new RanPark(lmp, seed);
  
  //Reneighboring//
  force_reneighbor = 1; //fix requires reneighboring
  next_reneighbor = -1; //timestep when reneigboring to be done, set to current step later in the code
  //global_freq = 1;
  comm_forward = 4; //Flag for comm size needed by this fix, might not be needed.
  
  //Fix per-atom vector for storing molstate//
  peratom_flag = 1;
  size_peratom_cols = 0;
  peratom_freq = 1;
  vector_atom = atom_state;

  //Output Info//
  if(me==0)
    {
      printf("Fix ucg info ...\n");
      printf("Total number of states %d\n",nstates_total);
      printf("Input rate matrix\n");
      for(int i=0;i<nstates_total;i++)
	{
	  for(int j=0;j<nstates_total;j++)
	    printf("%f ",rates[i][j]);
	  printf("\n");
	}
      printf("Input mhcorr matrix\n");
      for(int i=0;i<nstates_total;i++)
        {
          for(int j=0;j<nstates_total;j++)
            printf("%f ",mhcorr[i][j]);
          printf("\n");
        }
      printf("Total molecules calc. by UCG %d, total defined in input datafile  %d molecules\n\n",nmolreal,nmol);
    }
}

/* ---------------------------------------------------------------------- */
FixUCGStateTrans3_3_1::~FixUCGStateTrans3_3_1()
{
  delete random;
  delete randomp;
  //delete temperature if fix created it, similar to fix_langevin
  if (tcomputeflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  if(me==0)
    {
      //print initialstate.in file for restart purposes
    }
  
  if(allocated)
    {
      memory->destroy(nspecies_beads);
      memory->destroy(mol_endid);
      memory->destroy(mol_species);
      memory->destroy(atomtype_offset);
      memory->destroy(bondtype_offset);
      memory->destroy(nstates);
      memory->destroy(trans_flag);
      memory->destroy(mol_state);
      memory->destroy(mol_desum);
      memory->destroy(mol_accept);

      if(me==0)
        {
	  memory->destroy(reax_dir);
	  memory->destroy(rates);
	  memory->destroy(mhcorr);
	  memory->destroy(mol_desum_global);
	  memory->destroy(restrictmol);
        }
      memory->destroy(atom_state);
    }
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::allocate()
{
  if(!allocated)
    {
      int nmolp1 = nmol+1;
      nmax = atom->nmax;
      allocated = true;
     
      memory->create(nspecies_beads, nspecies, "fix:nspecies_beads");
      memory->create(mol_endid, nmolp1, "fix:mol_endid");
      memory->create(mol_species,nmolp1,"fix:molspecies");
      memory->create(atomtype_offset,nspecies,"fix:atomtype_offset");
      memory->create(bondtype_offset,nspecies,"fix:bondtype_offset");
      memory->create(nstates,nspecies,"fix:nstates");
      memory->create(trans_flag,nmolp1,"fix:transflag");
      memory->create(mol_state,nmolp1,"fix:molstate");
      memory->create(mol_desum,nmolp1,"fix:moldesum");
      memory->create(mol_accept,nmolp1,"fix:molaccept");
     
      if(me==0)
        {
	  memory->create(reax_dir,nstates_total,nstates_total,"fix:reax_dir");
	  memory->create(rates,nstates_total,nstates_total,"fix:rates");
	  memory->create(mhcorr,nstates_total,nstates_total,"fix:mhcorr");
	  memory->create(mol_desum_global,nmolp1,"fix:moldesumglobal");
	  memory->create(restrictmol,nmolp1,"fix:restrictmol");
        }
      memory->create(atom_state,nmax,"fix:atomstate");
    }
}

/* ---------------------------------------------------------------------- */
int FixUCGStateTrans3_3_1::setmask()
{
  int mask=0;
  mask |= END_OF_STEP;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/*------------------------------------------------------------------------*/
void FixUCGStateTrans3_3_1::consistencycheck()
{
  /* check if atom_types and bond_types are consistent with nstates */
  {
    int nlocal = atom->nlocal;
    int *type = atom->type;
    int *mask = atom->mask;
    tagint *tag = atom->tag;
    tagint *molecule = atom->molecule;
    int *num_bond = atom->num_bond;
    int **bond_type = atom->bond_type;

    int ntypes_expected=0,nbondtypes_expected=0;
    for(int i=0;i<nspecies;i++)
      {
	ntypes_expected+=ntypes_expected+nstates[i]*atomtype_offset[i];
	nbondtypes_expected+=nbondtypes_expected+nstates[i]*bondtype_offset[i];
      }
    
    if((atom->ntypes!=ntypes_expected) || (atom->nbondtypes!=nbondtypes_expected)) error->all(FLERR,"certain atom or bond types undefined. check force-field or fix ucg input");
    
    if(nmol!=nmolreal)
      error->all(FLERR,"mismatch in total molecules calculated by fix ucg. check initialstatefile for negative states");
      
    //can possibly add a condition to check if atom types and bond types of the molecules are consistent with their defined initialstate 
  }
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::init()
{
  if(strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
  
  //need a full neighbor list, built whenever re-neighboring occurs//
  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix  = 1;
  neighbor->requests[irequest]->half = 0; //default is half list
  neighbor->requests[irequest]->full = 1;
  
  int icompute = modify->find_compute(id_temp);
  if(icompute<0) error->all(FLERR,"Temperature ID for fix ucg does not exist");
  temperature = modify->compute[icompute];
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::setup(int vflag)
{
  //compute temperature before first call to post_force
  end_of_step();
  consistencycheck();
  if(strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else
    {
      ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
      post_force_respa(vflag,nlevels_respa-1,0);
      ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
    }
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::post_force_respa(int vflag, int ilevel, int iloop)
{
  if(ilevel == nlevels_respa-1)
    post_force(vflag);
}
/* ---------------------------------------------------------------------- */

void FixUCGStateTrans3_3_1::normalize(double *a)
{
  double mag=0.0;
  for(int k=0;k<3;k++)
    mag += a[k]*a[k];
  if(mag<=0)
    {error->all(FLERR,"error in normalizing vector");}
  mag=sqrt(mag);
  for(int k=0;k<3;k++)
    a[k] /= mag;
}
/* --------------------------------------------------------------------- */

void FixUCGStateTrans3_3_1::computecross(double *a, double *b1, double *b2)
{
  a[0] = b1[1]*b2[2] - b1[2]*b2[1];
  a[1] = b1[2]*b2[0] - b1[0]*b2[2];
  a[2] = b1[0]*b2[1] - b1[1]*b2[0];
}
/* --------------------------------------------------------------------- */

double FixUCGStateTrans3_3_1::computedot(double *b1, double *b2)
{
  return(b1[0]*b2[0]+b1[1]*b2[1]+b1[2]*b2[2]);
}

/* ---------------------------------------------------------------------- */

double FixUCGStateTrans3_3_1::computedihedral(double *x1, double *x2, double *x3, double *x4)
{
  double b1[3],b2[3],b3[3];
  double b1sq,b2sq,b3sq;
  b1sq = b2sq = b3sq = 0.0;
  for(int k=0;k<3;k++)
    {
      b1[k] = x1[k]-x2[k];
      b2[k] = x2[k]-x3[k];
      b3[k] = x3[k]-x4[k];
      //domain->minimum_image(b1[0],b1[1],b1[2]);
      //domain->minimum_image(b2[0],b2[1],b2[2]);
      //domain->minimum_image(b3[0],b3[1],b3[2]);
    }
  normalize(&b1[0]);
  normalize(&b2[0]);
  normalize(&b3[0]);

  double a[3],b[3];
  
  computecross(&a[0],b1,b2);
  computecross(&b[0],b2,b3);

  normalize(&a[0]);
  normalize(&b[0]);
  
  double cosphi;
  cosphi = 0.0;
  for(int k=0;k<3;k++)
    {
      cosphi -= a[k]*b[k];
    }
  return(cosphi);
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::rotatepointalongaxisbyangle(double *q, double *p, double theta, double *r)
{
  /** http://paulbourke.net/geometry/rotate/source.c
      Rotate a point p by angle theta around an arbitrary axis r
      Return the rotated point.
      Positive angles are anticlockwise looking down the axis
      towards the origin.
      Assume right hand coordinate system.
      Output is in q
  */
  double q1[3];
  q1[0] = q1[1] = q1[2] = 0.0;
  double costheta,sintheta;

  normalize(&r[0]);
  costheta = cos(theta);
  sintheta = sin(theta);

  q1[0] += (costheta + (1 - costheta) * r[0] * r[0]) * p[0];
  q1[0] += ((1 - costheta) * r[0] * r[1] - r[2] * sintheta) * p[1];
  q1[0] += ((1 - costheta) * r[0] * r[2] + r[1] * sintheta) * p[2];

  q1[1] += ((1 - costheta) * r[0] * r[1] + r[2] * sintheta) * p[0];
  q1[1] += (costheta + (1 - costheta) * r[1] * r[1]) * p[1];
  q1[1] += ((1 - costheta) * r[1] * r[2] - r[0] * sintheta) * p[2];

  q1[2] += ((1 - costheta) * r[0] * r[2] - r[1] * sintheta) * p[0];
  q1[2] += ((1 - costheta) * r[1] * r[2] + r[0] * sintheta) * p[1];
  q1[2] += (costheta + (1 - costheta) * r[2] * r[2]) * p[2];

  q[0] = q1[0]; q[1] = q1[1]; q[2] = q1[2];
}
/* ---------------------------------------------------------------------- */

void FixUCGStateTrans3_3_1::computemoldesum()
{
  int nmolp1=nmol+1;
  double fforce[2],bond_fforce[2]; 
  double factor_lj, factor_coul;
  factor_lj = factor_coul = 1.0; 
  
  //Neighbor list variables//
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int jnum;
  int *jlist;
  
  //bond list variables//
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  
  //per atom variables//
  double **x = atom->x;
  double *special_lj = force->special_lj;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  const bigint natoms = atom->natoms;
  
  //delU for pair interactions
  {
    for(int ii=0;ii<inum;ii++)
      {
	int i = ilist[ii];
	if(!(mask[i] & groupbit)) continue; //check atom 'i' is in group.
	int imol = molecule[i];
	int ispecies=mol_species[imol];
	if(trans_flag[imol]==0) continue;
	int itype = type[i];
	int newitype = getnewatomtype(tag[i],imol,itype,ispecies,0);
       
	jlist = firstneigh[i];
	jnum = numneigh[i];
	for(int jj=0;jj<jnum;jj++)
	  {
	    int j = jlist[jj];
	    factor_lj = special_lj[sbmask(j)]; //Is set to 0 if ignoring pair energies between bonded atoms
	    j &= NEIGHMASK;
	    if(!(mask[j] & groupbit)) continue; //check atom 'j' is in group
	    int jmol = molecule[j];
	    int jspecies=mol_species[jmol];
	    int jtype = type[j];
	    int newjtype;

	    if(imol==jmol && tag[i]>tag[j]) continue; //skip j>i to avoid double counting
	    if(trans_flag[jmol]==0 || imol != jmol) newjtype = jtype; //keep old jtype atom if 'j' belongs to different molecule irrespective of it is selected for transition
	    else 
		newjtype = getnewatomtype(tag[j],jmol,jtype,jspecies,0); //this condition executes when atom 'j' belongs to same molecule as atom 'i' and atom ID of 'i' is less than atom ID of 'j'

	    double dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2];
	    double rsq = dx*dx+dy*dy+dz*dz;
	    if(rsq<cutsq[itype][jtype])
	      {
		double energy_old = force->pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fforce[0]);
		double energy_new = force->pair->single(i,j,newitype,newjtype,rsq,factor_coul,factor_lj,fforce[1]);
		mol_desum[imol] += energy_new-energy_old;	
	      } 
	  }
      }
  }

  //delU for bond interactions if any
  {
    for(int n=0;n<nbondlist;n++)
      {
	int i = bondlist[n][0];
	int inext = bondlist[n][1];
	int ibondtype = bondlist[n][2];
	int imol = molecule[i];
	int ispecies=mol_species[imol];
	if(trans_flag[imol]==0) continue;
	int newibondtype = ibondtype + trans_flag[imol]*bondtype_offset[ispecies];
	double dx = x[i][0]-x[inext][0], dy = x[i][1]-x[inext][1], dz = x[i][2]-x[inext][2];
	double rsq = dx*dx+dy*dy+dz*dz;
	double energy_old = force->bond->single(ibondtype,rsq,i,inext,bond_fforce[0]);
	double energy_new = force->bond->single(newibondtype,rsq,i,inext,bond_fforce[1]);
	mol_desum[imol] += energy_new-energy_old;
      }
  }

  //Sum all the energy differences to get the total energy difference for each mol. Have to Bcast if all processor checks for reactions separately
  MPI_Reduce(&mol_desum[0],&mol_desum_global[0],nmolp1,MPI_DOUBLE,MPI_SUM,0,world);
} 

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::post_force(int vflag)
{
  //VARIABLES//
  unsigned short int change_flag=0,accept_flag=0;
  double rand,beta,mhterm;
  int nmolp1=nmol+1;
  double fforce[2],bond_fforce[2]; //value not used
  double factor_lj, factor_coul;
  factor_lj = factor_coul = 1.0; 

  //temperature
  {
    //t_current = temperature->compute_scalar();
    beta = 1.0/(t_current * force->boltz);
  }
  
  //Neighbor list variables//
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int jnum;
  int *jlist;
  
  //bond list variables//
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  
  //per atom variables//
  double **x = atom->x;
  double *special_lj = force->special_lj;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  const bigint natoms = atom->natoms;
  imageint *image = atom->image;

  // reallocate work arrays if necessary
  if (atom->nmax > nmax) {
    memory->destroy(atom_state);
    nmax = atom->nmax;
    memory->create(atom_state,nmax,"fix:atomstate");
    vector_atom = atom_state;
  }
  for(int i=0;i<nlocal;i++) {int imol=molecule[i]; atom_state[i]=mol_state[imol];}
  
  /*---------------------------------------------------------------------------------------------------*/
  /* The following steps are implemented for UCG transitions                                           */
  /* The UCG state transitions are governed by equation 8 and 9 in DOI:dx.doi.org/10.1021/ct500834     */
  /* Processor 0 loops over all molecules and selects them for reaction based on input 'k' values      */
  /* The energy difference of each of the selected molecules is computed                               */
  /* Processor 0 loops over all molecules and executes Metropolis-Hasting criterion to accept reaction */
  /* If reactions accepted, atom types/bond types are updated                                          */
  /*---------------------------------------------------------------------------------------------------*/

  /* step 1: variables that keep track of molecules selected for reaction reset to zero every step */
  reset_accumulators(); 

  /* step 2: Processor 0 select molecules for reaction based on inputs */
  {
    if(me==0)
      {
	change_flag = 0;
	double rsum;
	for(int imol=1; imol<=nmolreal; imol++)
	  {
	    if(restrictmolflag==1)
	      if(restrictmol[imol]==0) continue;
	    
	    rsum = 0.0;
	    short int istate = mol_state[imol];
	    rand = randomp->uniform();		

	    for(int j=0;j<nstates_total;j++)
	      {
		double rate=rates[istate][j];

		if (istate==j) continue;	       
	      
		rsum += rate;
		if(rand<rsum)
		  {
		    trans_flag[imol] = j-istate; 
		    change_flag = 1; 
#ifdef DIAGNOSTICS
		    printf("UCG selected mol %d with state %d direction %d at step %ld with rate %f %f and rand %f\n",imol,istate,trans_flag[imol],(long)update->ntimestep,rsum,rate,rand);
#endif
		    break;		    
		  }
	      }
	  }
      }
    MPI_Bcast(&change_flag,1,MPI_UNSIGNED_SHORT,0,world);
  }

#ifdef DIAGNOSTICS
  if(me==0)
    {
      std::ofstream write("mol_transflag");
      write<<"trans_flag"<<' '<<"mol_filamentID_poly"<<' '<<"mol_filamentID_depoly"<<"\n";
      for (int imol=1;imol<=nmol;imol++)
	write<<imol<<' '<<trans_flag[imol]<<' '<<mol_filamentID_poly[imol]<<' '<<mol_filamentID_depoly[imol]<<"\n";
      write.close();
    }
#endif
   
  /* step 3 and 4: if reactions were selected, calculate energy difference for the reaction and execute Metropolis-Hastings criterion to decide if selected reactions are to be accepted */
  if(change_flag==1)
    {
      MPI_Bcast(&trans_flag[0],nmolp1,MPI_SHORT,0,world); //needed for computemoldesum function
      computemoldesum();
      
      /* Meteropolis-Hastings acceptance criterion by Processor 0  only */
      if(me==0)
        {
	  accept_flag=0;
	  for(int imol=1;imol<=nmolreal;imol++)
            {
	      int istate=mol_state[imol];
	      if(trans_flag[imol]==0) continue;
	      double detailedbalancefactor=1.0;
	      detailedbalancefactor=(rates[istate+trans_flag[imol]][istate])/(rates[istate][istate+trans_flag[imol]]);  //R1->2 = k2->1/k1->2
	      rand = randomp->uniform();	
	      mhterm = MIN(detailedbalancefactor*exp(-Beta*mol_desum_global[imol] + mhcorr[istate][istate+trans_flag[imol]]), 1.0);
	     
	      if(rand<mhterm || mhterm==1.0)
                {
		  mol_accept[imol] = 1;
		  mol_state[imol] += trans_flag[imol]; 
		  accept_flag = 1;
		  printf("U m %d s %d -> %d s s %ld dU %f beta %f\n",imol,mol_state[imol]-trans_flag[imol],mol_state[imol],(long)update->ntimestep,mol_desum_global[imol],Beta);
                }
#ifdef MYDEBUG
	      else
		  printf("U m %d s %d -> %d f s %ld dU %f beta %f\n",imol,mol_state[imol],mol_state[imol]+trans_flag[imol],(long)update->ntimestep,mol_desum_global[imol],Beta);
#endif
            }
        }//me==0 if
      
      MPI_Bcast(&accept_flag,1,MPI_UNSIGNED_SHORT,0,world); //needed to check if atomtypes/bondtypes have to be updated
      
      /* step 5: check and update atomtypes/bondtypes */
      if(accept_flag==1)
        {
	  next_reneighbor = update->ntimestep; //Trigger reneighboring at the next timestep. This is required since atom types are updated here.//
	  MPI_Bcast(&mol_accept[0],nmol+1,MPI_UNSIGNED_SHORT,0,world);
	  MPI_Bcast(&mol_state[0],nmol+1,MPI_SHORT,0,world); 

	  set_filamentID(1,(long)update->ntimestep,Beta);

	 
#ifdef DIAGNOSTICS 
	  if(me==0)
            {
	      std::ofstream write("mol_state_accept");
	      write<<"molID"<<' '<<"mol_state"<<' '<<"mol_accept"<<"\n";
	      for(int imol=1;imol<nmol+1;imol++)
		write<<imol<<' '<<mol_state[imol]<<' '<<mol_accept[imol]<<"\n";
	      write.close();
            }
#endif
	  
	  //Atom types update//
	  for(int i=0;i<nlocal;i++)
            {
	      if(!(mask[i] & groupbit)) continue;
	      int imol = molecule[i];
	      atom_state[i] = mol_state[imol];
	      if(mol_accept[imol]==0) continue;
	      int itype = type[i];
	      int ispecies=mol_species[imol];
	      int newitype=getnewatomtype(tag[i],imol,itype,ispecies,1);
	      type[i]=newitype;

	      /* Bond type update: not using bondlist here,because neigh_bond.cpp creates bondlist using bond_atom and bond_type. Hence, need to overwrite/update bond_type and trigger build_topology */
	      for(int ibond=0;ibond<num_bond[i];ibond++)
                {
                  int inext = atom->map(bond_atom[i][ibond]);
                  //if(inext >= nlocal || inext < 0) continue; //Not using this because bond list is unique. i.e. if 1-2 exists, 2-1 does not
                  if(molecule[inext] != imol) error->all(FLERR,"Inter-molecule bond in fix ucg_state_trans2");
                  int ibondtype = bond_type[i][ibond];
                  int newibondtype = ibondtype + trans_flag[imol]*bondtype_offset[ispecies];
                  bond_type[i][ibond] = newibondtype;
                }
            }
	  neighbor->build_topology(); //Force rebuiliding of topology since bond parameters are updated
        }//accept_flag if
    }//change_flag if
  if(nprocs > 1) comm->forward_comm_fix(this);
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::end_of_step()
{
  t_current = temperature->compute_scalar();
}

/* ---------------------------------------------------------------------- */
int FixUCGStateTrans3_3_1::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int *type = atom->type;
  double **x = atom->x;
  int m=0,k;
  for (int i=0;i<n;i++)
    {
      k=list[i];
      buf[m++] = type[k];
      buf[m++] = x[k][0];
      buf[m++] = x[k][1];
      buf[m++] = x[k][2];

    }
  return m;
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::unpack_forward_comm(int n, int first, double *buf)
{
  int last=first+n;
  int m=0;
  int *type = atom->type;
  double **x = atom->x;
  for(int i=first;i<last;i++)
    {
      type[i] = static_cast<int> (buf[m++]);
      x[i][0] = buf[m++];
      x[i][1] = buf[m++];
      x[i][2] = buf[m++];
    }
}

/* ---------------------------------------------------------------------- */
double FixUCGStateTrans3_3_1::memory_usage()
{
  return 0;
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::reset_accumulators()
{
  int nmolp1 = nmol+1;
  for(int imol=0;imol<nmolp1;imol++)
    {
      trans_flag[imol]=0;
      mol_desum[imol]=0;
      mol_accept[imol]=0;
}
/*------------------------------------------------------------------------*/
int FixUCGStateTrans3_3_1::getnewatomtype(int id, int imol, int itype, int ispecies, int update)
{
    return(itype+trans_flag[imol]*atomtype_offset[ispecies]);
} 



