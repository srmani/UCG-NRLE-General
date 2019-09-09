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

///////
#include "random_park.h" // delete later
//////

#include <fstream> //read input files: rates and mhcorr
#include <string> //read input files: rates and mhcorr
#include <cmath> //read input files: rates and mhcorr
using namespace LAMMPS_NS;
using namespace FixConst; //in fix.h, defines POST_FORCE, etc.

#define MYDEBUG
//#define MYDEBUG1
#define PI 3.14159
/* ---------------------------------------------------------------------- */
FixUCGStateTrans3_3_1::FixUCGStateTrans3_3_1(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp,narg,arg)
{
  //Current usage: fix 1[0] all[1] ucg_state_trans2[2] 12121[3] nstates[4] 3[5] rates[6] rates.txt[7]  mhcorr[8] mhcorr.txt[9]
  // where rates for a given state i are the entries in column i of rates.txt (1st row is forward rate, 2nd row is reverse rate). Same with mhcorr.
  
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  
  int iarg=0;
  iarg++; // fix-id [0]
  iarg++; // group [1]
  iarg++; // fix-name [2]
  int seed = force->inumeric(FLERR,arg[iarg++]); // seed [3]

  if(strcmp(arg[iarg++],"Beta")!=0) error->all(FLERR,"Illegal Beta in fix ucg_state_trans3 command"); //keyword-nspecies[4]
  Beta = force->numeric(FLERR,arg[iarg++]); // nspecies [5]

  
  if(strcmp(arg[iarg++],"nspecies")!=0) error->all(FLERR,"Illegal nspecies in fix ucg_state_trans3 command"); //keyword-nspecies[4]
  nspecies = force->inumeric(FLERR,arg[iarg++]); // nspecies [5]
  
  if(strcmp(arg[iarg++],"nstates")!=0) error->all(FLERR,"Illegal nstates in fix ucg_state_trans3 command"); //keyword-nstates[4]
  int *nstates_temp=new int [nspecies];
  nstates_total=0;
  for (int i=0;i<nspecies;i++){
    nstates_temp[i]=force->inumeric(FLERR,arg[iarg++]);
    nstates_total+=nstates_temp[i];
  }

  if (strcmp(arg[iarg++], "speciesbeads") != 0) error->all(FLERR, "Illegal species beads in fix ucg_state_trans3 command"); //keyword-nstates[4]
  int *nspecies_beads_temp = new int[nspecies];
  for (int i = 0; i<nspecies; i++) {
    nspecies_beads_temp[i] = force->inumeric(FLERR, arg[iarg++]);
  }
  
  if(strcmp(arg[iarg++],"aoffset")!=0) error->all(FLERR,"Illegal atomtypeoffset in fix ucg_state_trans3 command"); //keyword-nstates[4]
  int *aoffset_temp=new int[nspecies];
  for(int i=0;i<nspecies;i++){
    aoffset_temp[i]=force->inumeric(FLERR,arg[iarg++]);
  }

  if(strcmp(arg[iarg++],"boffset")!=0) error->all(FLERR,"Illegal bondtypeoffset in fix ucg_state_trans3 command"); //keyword-nstates[4]
  int *boffset_temp=new int[nspecies];
  for(int i=0;i<nspecies;i++){
    boffset_temp[i]=force->inumeric(FLERR,arg[iarg++]);
  }
 
  if(strcmp(arg[iarg++],"polymerizationflag")!=0) error->all(FLERR,"Illegal polymerization flag in fix ucg_state_trans2 command"); //keyword-nstates[4]
  polymerizationflag = force->inumeric(FLERR,arg[iarg++]);

  if(strcmp(arg[iarg++],"nfilaments")!=0) error->all(FLERR,"Illegal number of filaments in fix ucg_state_trans2 command"); //keyword-nstates[4]
  nfilaments=force->inumeric(FLERR,arg[iarg++]);

  

  //Find nmol and check data file consistency//
  {
    //Find number of molecules//
    int nlocal = atom->nlocal;
    tagint *molecule = atom->molecule;	
    int maxmol_local=-1;
    for(int i=0;i<nlocal;i++) {if(molecule[i]>maxmol_local) maxmol_local=molecule[i];}
    MPI_Allreduce(&maxmol_local,&nmol,1,MPI_INT,MPI_MAX,world); //nmol
  }
    
  //Allocate arrays//
  allocated=false;  allocate();

  for(int i=0;i<nspecies;i++){
    nstates[i]=nstates_temp[i];
  }
  delete [] nstates_temp;

  for (int i = 0; i<nspecies; i++) {
    nspecies_beads[i] = nspecies_beads_temp[i];
  }
  delete[] nspecies_beads_temp;
  ntypereal=0;
  for(int i=0;i<nspecies;i++){
    atomtype_offset[i]=aoffset_temp[i];
    ntypereal+=atomtype_offset[i]*nstates[i];
  }
  delete [] aoffset_temp;

  for(int i=0;i<nspecies;i++){
    bondtype_offset[i]=boffset_temp[i];
  }
  delete [] boffset_temp;

  for(int imol=0;imol<(nmol+1);imol++)
    {
      interaction_factor[imol]=1;
      timer[imol]=0;
    }


  //Read rates and mhcorr terms//
  if(me==0)
    {
      for(int i=0;i<nstates_total;i++){for(int j=0;j<nstates_total;j++) rates[i][j] = mhcorr[i][j] = 0.0;} //default rates and mhcorr
      while(iarg<narg)
        {
	  if(strcmp(arg[iarg],"filamentsinfo")==0)
	    {
	      iarg++;
	      if(narg<iarg+2) error->one(FLERR,"Illegal fix ucg_state_trans2 command");
	      //read filament info files.
	      {
		std::ifstream filendsfile(arg[iarg++]);
		if(!filendsfile) {error->one(FLERR, "filament ends file missing in fix ucg_state_trans3 command");}

		std::ifstream filidentityfile(arg[iarg++]);
		if(!filidentityfile) {error->one(FLERR, "filament identity file missing in fix ucg_state_trans3 command");}
		
		int temp, bend, pend;
		filament_ends[0][0]=0;filament_ends[0][1]=0;
		for(int filcount=1;filcount<(nfilaments+1);filcount++)
		  {
		    filendsfile>>temp>>bend>>pend;
		    filament_ends[filcount][0]=bend;
		    filament_ends[filcount][1]=pend;
		  }
		filendsfile.close();
		
		int molid, molfil, molfilfor, molfilback;
		for(int imol=1;imol<(nmol+1);imol++)
		  {
		    filidentityfile>>molid>>molfil>>molfilfor>>molfilback;
		    mol_filamentID[imol]=molfil;
		    mol_filamentlistforward[imol]=molfilfor;
		    mol_filamentlistbackward[imol]=molfilback;
		  }
		filidentityfile.close();
	      }
	    }

	  else if(strcmp(arg[iarg],"rates")==0)//rates
            {
	      iarg++;
	      if(narg<iarg+1) error->one(FLERR,"Illegal fix ucg_state_trans2 command");
	      //read_rates(arg[iarg++]);
	      {
		double temprate;
		std::ifstream ratesfile(arg[iarg++]);
		if(!ratesfile) {error->one(FLERR,"Rates file missing in fix ucg_state_trans2 command");}
		for(int i=0;i<nstates_total;i++)
		  for(int j=0;j<nstates_total;j++)
		    { ratesfile>>temprate;
		      //rates[i][j] = force->numeric(FLERR,tempchar);//FIX THIS, HAVE TO CHECK IF RATE IS A NUMBER
		      rates[i][j] = temprate;
		      if(rates[i][j]>1.0 || rates[i][j]<0) error->all(FLERR,"Illegal rates in fix ucg_state_trans3 command");} //read rates
		ratesfile.close();
	      }
            }

	  else if(strcmp(arg[iarg],"mhcorr")==0)//mhcorr
            {
	      iarg++;
	      if(narg<iarg+1) error->one(FLERR,"Illegal fix ucg_state_trans2 command");
	      //read_mhcorr
	      {
		double tempmhcorr;
		std::ifstream mhcorrfile(arg[iarg++]);
		if(!mhcorrfile) {error->one(FLERR,"MHcorr file missing in fix ucg_state_trans2 command");}
		for(int i=0;i<nstates_total;i++)
		  for(int j=0;j<nstates_total;j++)
		    { mhcorrfile>>tempmhcorr;
		      //mhcorr[i][j] = force->numeric(FLERR,tempmhcorr);//FIX THIS, HAVE TO CHECK IF MHCORR IS A NUMBER
		      mhcorr[i][j]=tempmhcorr;
		    }
		mhcorrfile.close();
	      }
            }
	  
	  else if(strcmp(arg[iarg],"dihedraldependence")==0)
            {
	      iarg++;
	      if(narg<iarg+2) error->one(FLERR,"Illegal fix ucg_state_trans2 command");
	      //read dihedral parameters file
	      {
		double tempphi0;

		std::ifstream dihedralphifile(arg[iarg++]);
		if (!dihedralphifile) { error->one(FLERR, "Dihedral parameter file missing in fix ucg_state_trans3 command"); }

		std::ifstream dihedraletafile(arg[iarg++]);
		if (!dihedraletafile) { error->one(FLERR, "Dihedral parameter file missing in fix ucg_state_trans3 command"); }

		for (int i = 0; i < nstates_total; i++)
		  for (int j = 0; j < nstates_total; j++)
		    {
		      dihedralphifile >> tempphi0;
		      //phi0[i][j] = force->numeric(FLERR,tempphi0);//FIX THIS, HAVE TO CHECK IF MHCORR IS A NUMBER
		      phi0[i][j] = tempphi0;
		    }
		for (int i = 0; i < nstates_total; i++)
		  for (int j = 0; j < nstates_total; j++)
		    {
		      dihedraletafile >> tempphi0;
		      //phi0[i][j] = force->numeric(FLERR,tempphi0);//FIX THIS, HAVE TO CHECK IF MHCORR IS A NUMBER
		      eta[i][j] = tempphi0;
		    }
		dihedralphifile.close();
		dihedraletafile.close();
	      }
            }

	  else if (strcmp(arg[iarg], "initialstatefile") == 0)
            {
	      iarg++;
	      if (narg < iarg + 1)  error->one(FLERR, "Illegal fix ucg_state_trans3 command");
	      //read initial state file and assign states
	      {
		std::ifstream istatefile(arg[iarg++]);
		if (!istatefile) { error->one(FLERR, "Initial state file missing in fix ucg_state_trans3 command"); }
		std::string istate0;
		char *istate1;
		int cumul_beads = 0;
		nmolreal=0;
		for (int imol = 1; imol < nmol + 1; imol++)
		  {
		    //Have to check if the number of lines in this file match the number of molecules. Also, check if input is an int.
		    istatefile >> istate0;
		    istate1 = (char *)alloca(istate0.size() + 1);
		    memcpy(istate1, istate0.c_str(), istate0.size() + 1);
		    mol_state[imol] = force->inumeric(FLERR, istate1);//FIX THIS, HAVE TO CHECK IF MHCORR IS A NUMBER
		    if(mol_state[imol]>=0) 
		      nmolreal++;
		    
		    int virtualflag=0;
		    int cumul_states = 0;
		    for (int ispecies = 0; ispecies < nspecies; ispecies++)
		      {
			cumul_states += nstates[ispecies];
			if(mol_state[imol]<0)
			  {
			    virtualflag=1;
			    mol_species[imol]=nspecies;
			    break;
			  }
			if (mol_state[imol] < cumul_states)
			  {
			    mol_species[imol] = ispecies;
			    break;
			  }
		      }
		    cumul_beads = cumul_beads + !virtualflag*nspecies_beads[mol_species[imol]] + virtualflag*12;
		    mol_endid[imol] = cumul_beads;
		  }
		istatefile.close();
	      }
#ifdef MYDEBUG1 
	      //////to delete later
	      std::ofstream write("mol_info_init");
	      write<<"molID"<<' '<<"mol_state"<<' '<<"mol_species"<<' '<<"mol_endid"<<"\n";
	      for(int imol=1;imol<nmol+1;imol++)
		write<<imol<<' '<<mol_state[imol]<<' '<<mol_species[imol]<<' '<<mol_endid[imol]<<"\n";
	      write.close();
	      //////to delete later
#endif
            }

	  else if (strcmp(arg[iarg], "restrictmol") == 0)
            {
	      iarg++;
	      if (narg < iarg + 1) error->one(FLERR, "Illegal fix ucg_state_trans2 command");
	      //read restrictmol file
	      {
		restrictmolflag = 1;
		std::ifstream restrictmolfile(arg[iarg++]);
		std::string temprestrictmol;
		char *temprestrictmol1;
		if (!restrictmolfile) { error->one(FLERR, "Restrict mol file missing in fix ucg_state_trans3 command"); }
		for (int imol = 1; imol < nmol + 1; imol++)
		  {
		    restrictmolfile >> temprestrictmol;
		    //phi0[i][j] = force->numeric(FLERR,tempphi0);//FIX THIS, HAVE TO CHECK IF MHCORR IS A NUMBER
		    temprestrictmol1 = (char *)alloca(temprestrictmol.size() + 1);
		    memcpy(temprestrictmol1, temprestrictmol.c_str(), temprestrictmol.size() + 1);
		    restrictmol[imol] = force->inumeric(FLERR, temprestrictmol1);
		  }
		restrictmolfile.close();
	      }
            }

	  else if (strcmp(arg[iarg], "direction") == 0) //forward backward reaction definition
            {
	      iarg++;
	      if (narg < iarg + 1) error->one(FLERR, "Illegal fix ucg_state_trans2 command");
	      {
		int Val;
		std::ifstream dirfile(arg[iarg++]);
		if (!dirfile) { error->one(FLERR, "Direction file missing in fix ucg_state_trans2 command"); }
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

	  else if (strcmp(arg[iarg],"scaling") == 0) // to account for rate differences along the filament. Can be used for rxns other that poly/depoly.
            {
	      iarg++;
	      if (narg < iarg + 1) error->one(FLERR, "Illegal fix ucg_state_trans2 command");
	      {
		double scalval;
		std::ifstream scalingfile(arg[iarg++]);
		if (!scalingfile) { error->one(FLERR, "scaling file missing in fix ucg_state_trans2 command"); }
		for (int i = 0; i < nstates_total; i++) {
		  for (int j = 0; j < nstates_total; j++)
		    {
		      scalingfile >> scalval;
		      scaling[i][j] = scalval;
		    }
		}
		scalingfile.close();
	      }
            }

	  else
	    error->one(FLERR, "Illegal fix ucg_state_trans2 command"); //unknown keyword
        }

    }
	 
  MPI_Bcast(&mol_state[0],nmol+1,MPI_SHORT,0,world);
  MPI_Bcast(&mol_species[0],nmol+1,MPI_UNSIGNED_SHORT,0,world);
  MPI_Bcast(&mol_endid[0], nmol+1, MPI_UNSIGNED_LONG, 0, world);
  MPI_Bcast(&mol_filamentID[0], nmol+1, MPI_INT, 0, world);
  MPI_Bcast(&mol_filamentlistforward[0], nmol+1, MPI_INT, 0, world);
  MPI_Bcast(&mol_filamentlistbackward[0], nmol+1, MPI_INT, 0, world);
  MPI_Bcast(&filament_ends[0][0], (nfilaments+1)*2, MPI_INT, 0, world);
  MPI_Bcast(&nmolreal,1,MPI_INT,0,world);
 
  tagint *molecule = atom->molecule;
  for(int i=0;i<atom->nlocal;i++) {int imol=molecule[i]; atom_state[i]=mol_state[imol];}

  set_filamentID(0,0,0);

  /*for (int i = 0; i<atom->nlocal; i++)
    {
      int imol = molecule[i]; atom_filamentID[i] = mol_filamentID[imol];
    }*/

  //check for pair style with cutoff
  if (force->pair == NULL) error->all(FLERR,"ucg_state_trans2 fix requires a pair style");
  if (force->pair->cutsq == NULL) error->all(FLERR,"ucg_state_trans2 fix is incompatible with pair style");
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
 
  ///////
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
      printf("Fix ucg_state_trans2 info ...\n");
      printf("  number of states %d, number of molecules %d\n input rates matrix\n",nstates_total,nmol);
      
     for(int i=0;i<nstates_total;i++)
       {
	 for(int j=0;j<nstates_total;j++)
	   {
	     printf("%f\t",rates[i][j]);
	   }
	 printf("\n");
       }
     printf("\v\v Real %d of total %d molecules\n\n",nmolreal,nmol);
    }

  //printf("Exiting constructor %d\n",me);
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
      std::ofstream write("filament_ends_continue");
      for(int i=1;i<=nfilaments;i++)
	write<<i<<' '<<filament_ends[i][0]<<' '<<filament_ends[i][1]<<"\n";
      write.close();
      
      std::ofstream print("filament_identity_continue");
      for(int imol=1;imol<(nmol+1);imol++)
	print<<imol<<' '<<mol_filamentID[imol]<<' '<<mol_filamentlistforward[imol]<<' '<<mol_filamentlistbackward[imol]<<"\n";
      print.close();
    }
  
  if(allocated)
    {
      if(polymerizationflag)
        {
	  memory->destroy(xvirt);
	  memory->destroy(mol_filamentlistbackward);
	  memory->destroy(mol_filamentlistforward);
	  memory->destroy(mol_filamentID_poly);
	  memory->destroy(mol_filamentID_depoly);
	  memory->destroy(filament_endsdepolymflag);
	  memory->destroy(filament_endspolymflag);
	  memory->destroy(filament_ends);
	  memory->destroy(mol_filamentID);
	  memory->destroy(xpolym);
	  memory->destroy(xpolymtype);    
        }
      memory->destroy(interaction_factor);
      memory->destroy(timer);
      memory->destroy(nspecies_beads);
      //memory->destroy(atom_filamentID);
      memory->destroy(mol_endid);
      memory->destroy(mol_species);
      memory->destroy(atomtype_offset);
      memory->destroy(bondtype_offset);
      memory->destroy(nstates);
      memory->destroy(phi0);
      memory->destroy(eta);
      memory->destroy(trans_flag);
      memory->destroy(mol_state);
      memory->destroy(mol_desum);
      memory->destroy(mol_accept);
      memory->destroy(xdihed);
      if(me==0)
        {
	  memory->destroy(scaling);
	  memory->destroy(reax_dir);
	  memory->destroy(rates);
	  memory->destroy(mhcorr);
	  memory->destroy(mol_desum_global);
	  memory->destroy(mol_dihed);
	  memory->destroy(restrictmol);
        }
      memory->destroy(atom_state);
    }
  
  //printf("Exiting destructor %d\n",me);

}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::allocate()
{
  if(!allocated)
    {
      int nmolp1 = nmol+1;
      int nfilamentsp1=nfilaments+1;
      nmax = atom->nmax;
      allocated = true;
      if(polymerizationflag)
        {
	  memory->create(xvirt,nfilaments*2,12,3,"fix:xvirt");
	  memory->create(mol_filamentlistbackward, nmolp1, "fix:mol_filamentlistbackward");
	  memory->create(mol_filamentlistforward,nmolp1,"fix:mol_filamentlistforward");
	  memory->create(mol_filamentID_poly, nmolp1, "fix:mol_filamentID_poly");
	  memory->create(mol_filamentID_depoly, nmolp1, "fix:mol_filamentID_depoly");
	  memory->create(filament_endsdepolymflag,nfilamentsp1,2,"fix:filament_endsdepolymflag");
	  memory->create(filament_endspolymflag,nfilamentsp1,2,"fix:filaments_endspolymflag");
	  memory->create(filament_ends,nfilamentsp1,2,"fix:filaments_end");
	  memory->create(mol_filamentID,nmolp1,"fix:molfilamentID");
	  memory->create(xpolym,nmolp1,3,"fix:xpolym");
	  memory->create(xpolymtype, nmolp1, "fix:xpolymtype");
        }
      memory->create(interaction_factor,nmolp1,"fix:interaction_factor");
      memory->create(timer,nmolp1,"fix:timer");
      memory->create(nspecies_beads, nspecies, "fix:nspecies_beads");
      //memory->create(atom_filamentID,nmax,"fix:atomfilametID");
      memory->create(mol_endid, nmolp1, "fix:mol_endid");
      memory->create(mol_species,nmolp1,"fix:molspecies");
      memory->create(atomtype_offset,nspecies,"fix:atomtype_offset");
      memory->create(bondtype_offset,nspecies,"fix:bondtype_offset");
      memory->create(nstates,nspecies,"fix:nstates");
      memory->create(phi0,nstates_total,nstates_total,"fix:phi0");
      memory->create(eta,nstates_total,nstates_total,"fix:eta");
      memory->create(trans_flag,nmolp1,"fix:transflag");
      memory->create(mol_state,nmolp1,"fix:molstate");
      memory->create(mol_desum,nmolp1,"fix:moldesum");
      memory->create(mol_accept,nmolp1,"fix:molaccept");
      memory->create(xdihed,nmolp1,12,3,"fix:xdihed");
      if(me==0)
        {
	  memory->create(scaling,nstates_total,nstates_total,"fix:scaling");
	  memory->create(reax_dir,nstates_total,nstates_total,"fix:reax_dir");
	  memory->create(rates,nstates_total,nstates_total,"fix:rates");
	  memory->create(mhcorr,nstates_total,nstates_total,"fix:mhcorr");
	  memory->create(mol_desum_global,nmolp1,"fix:moldesumglobal");
	  memory->create(mol_dihed,nmolp1,"fix:dihed");
	  memory->create(restrictmol,nmolp1,"fix:restrictmol");
        }
      memory->create(atom_state,nmax,"fix:atomstate");
    }
  //printf("Exiting allocation %d\n",me);
}

/* ---------------------------------------------------------------------- */
int FixUCGStateTrans3_3_1::setmask()
{
  int mask=0;
  mask |= INITIAL_INTEGRATE;
  mask |= PRE_EXCHANGE;
  mask |= END_OF_STEP;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  return mask;
}

void FixUCGStateTrans3_3_1::consistencycheck()
{
  //check if atom_types and bond_types are consistent with nstates, and find barbedend_type//
  {
    int nlocal = atom->nlocal;
    int *type = atom->type;
    int *mask = atom->mask;
    tagint *tag = atom->tag;
    tagint *molecule = atom->molecule;
    int *num_bond = atom->num_bond;
    int **bond_type = atom->bond_type;
    if((atom->ntypes)%nstates_total!=0 || (atom->nbondtypes)%nstates_total !=0) error->all(FLERR,"Inconsistent atom/bond types in fix_ucg_state_trans3 command");
    
    for(int i=0;i<nstates_total;i++)
      {
	for(int j=0;j<nstates_total;j++)
	  {
	    if((i>=nstates[0] || j>=nstates[0]) && (eta[i][j]!=0))
	      error->all(FLERR,"Inconsistent eta input for non-actin species in fix_ucg_state_trans3 command");
          
	    /*for (int k=0;k<nspecies;k++)
	      {
	      if(i==nstates[k]-1 && rates[i][j]!=0 && j>=i)
                      
	      if(i==nstates[k]-1 && rates[i][j]!=0 && j>=i)
                      
	      if(i==nstates[k]-1 && rates[i][j]!=0 && j>=i)
                          
	      }*/
              
	  }
      }

    if(!polymerizationflag)
      {
	if(nmol!=nmolreal)
	  error->all(FLERR,"Inconsistent initialstate.in file. negative initial states for non-polymerization");
      }
    
      
    //check if initial types are for single state//
    /*int inconsistentflag=0;
      for(int i=0;i<nlocal;i++)
      {
      if(!(mask[i] & groupbit)) continue; // i belongs to the group
      int imol = molecule[i];
      int ispecies=mol_species[imol];
      if(type[i]<mol_state[imol]*atomtype_offset[ispecies] || type[i]>(mol_state[imol]+1)*atomtype_offset[ispecies]) {inconsistentflag=1;} //states go from 0 to nstates-1
	
      for(int ibond=0;ibond<num_bond[i];ibond++)
      if(bond_type[i][ibond]<mol_state[imol]*bondtypeoffset || bond_type[i][ibond]>(mol_state[imol]+1)*bondtypeoffset) {inconsistentflag=2;}
      }
      int inconsistentflag1;
      MPI_Reduce(&inconsistentflag,&inconsistentflag1,1,MPI_INT,MPI_SUM,0,world);
      if(me==0)
      {
      if(inconsistentflag1!=0)
      error->one(FLERR,"Inconsistent atom/bond type in fix_ucg_state_trans4 command");
      }*/
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
  if(icompute<0) error->all(FLERR,"Temperature ID for fix ucg_state_trans2 does not exist");
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
  //printf("Exiting setup %d\n",me);
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::post_force_respa(int vflag, int ilevel, int iloop)
{
  if(ilevel == nlevels_respa-1)
    post_force(vflag);
}

void FixUCGStateTrans3_3_1::normalize(double *a)
{
  double mag=0.0;
  for(int k=0;k<3;k++)
    mag += a[k]*a[k];
  if(mag<=0)
    {error->all(FLERR,"Bond length error in fix_ucg_state_trans3");}
  mag=sqrt(mag);
  for(int k=0;k<3;k++)
    a[k] /= mag;
}
void FixUCGStateTrans3_3_1::computecross(double *a, double *b1, double *b2)
{
  a[0] = b1[1]*b2[2] - b1[2]*b2[1];
  a[1] = b1[2]*b2[0] - b1[0]*b2[2];
  a[2] = b1[0]*b2[1] - b1[1]*b2[0];
}
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

void FixUCGStateTrans3_3_1::findvirtualsite(int ivirt, tagint endmol, bool signflag, int monodiff)
{
  // barbed end gets signflag=false, pointed end gets signflag=true
  double rise,twist;
  if(signflag==true){ rise = monodiff*27.59; twist = monodiff*2.904228;} else { rise = -monodiff*27.59; twist = -monodiff*2.904228; }

  double x1oda[3],x2oda[3],x3oda[3],x4oda[3],x1odashift[3],x2odashift[3],x3odashift[3];
  double xtrans[3],x1a[3],x2a[3],x2an[3],x3a[3],x4a[12][3],x1b[3],x2b[3],x2bn[3],x3b[3],x3bn[3],x4b[12][3],cross[3],cross2[3],zaxis[3],theta12,theta13,x3bperp[3],x3odashiftperp[3];

  /* Aligned Oda monomer coordinates for CG sites 1-3*/

  x1oda[0] = 25.061113357543945; x1oda[1] = 9.4896163940429690; x1oda[2] = -1.7848323583602905;
  x2oda[0] = 15.925351142883300; x2oda[1] = 7.6210808753967285; x2oda[2] = 18.4297752380371100;
  x3oda[0] = 12.024409294128418; x3oda[1] = -8.710589408874512; x3oda[2] = -13.981396675109863;
    
  for(int k=0;k<3;k++)
    {
      x1odashift[k] = x1oda[k] - x1oda[k];
      x2odashift[k] = x2oda[k] - x1oda[k];
      x3odashift[k] = x3oda[k] - x1oda[k];
    }
  normalize(&x2odashift[0]);
  normalize(&x3odashift[0]);
    
  //translation

  for(int k=0;k<3;k++)
    {
      xtrans[k]=xdihed[endmol][0][k]; //store translation for later use
        
      x1a[k]=xdihed[endmol][0][k]-xtrans[k]; //must be zero
      x2a[k]=xdihed[endmol][1][k]-xtrans[k];
      x3a[k]=xdihed[endmol][2][k]-xtrans[k];
      
      for(int l=3;l<12;l++)  
	x4a[l][k]=xdihed[endmol][l][k]-xtrans[k];
  
      x2an[k]=x2a[k];
    }

  normalize(&x2an[0]);
    
  computecross(&cross[0],&x2an[0],&x2odashift[0]);
  theta12=atan2(sqrt(computedot(&cross[0],&cross[0])),computedot(&x2an[0],&x2odashift[0]));
    

  //first rotation
    
  rotatepointalongaxisbyangle(&x2b[0],&x2a[0],theta12,&cross[0]);
  rotatepointalongaxisbyangle(&x3b[0],&x3a[0],theta12,&cross[0]);
  for(int l=3;l<12;l++)
    rotatepointalongaxisbyangle(&x4b[l][0],&x4a[l][0],theta12,&cross[0]);
  
  
  //second rotation
  for(int k=0;k<3;k++)
    x3bn[k] = x3b[k];
    
  normalize(&x3bn[0]);
  
  for(int k=0;k<3;k++)
    {
      x3bperp[k]=x3b[k]-computedot(&x3b[0],&x2odashift[0])*x2odashift[k];
      x3odashiftperp[k] = x3odashift[k]-computedot(&x3odashift[0],&x2odashift[0])*x2odashift[k];
    }
  
  normalize(&x3bperp[0]);
  normalize(&x3odashiftperp[0]);
  
  computecross(&cross2[0],&x3bperp[0],&x3odashiftperp[0]);
  theta13=atan2(sqrt(computedot(&cross2[0],&cross2[0])),computedot(&x3odashiftperp[0],&x3bperp[0]));

  rotatepointalongaxisbyangle(&x2a[0],&x2b[0],theta13,&cross2[0]); //must not change, within precision
  rotatepointalongaxisbyangle(&x3a[0],&x3b[0],theta13,&cross2[0]);
  for(int l=3;l<12;l++)
    rotatepointalongaxisbyangle(&x4a[l][0],&x4b[l][0],theta13,&cross2[0]); 
    
  
  //shift back by oda
  for(int k=0;k<3;k++)
    {
      x1a[k] += x1oda[k];
      x2a[k] += x1oda[k];
      x3a[k] += x1oda[k];
      for(int l=3;l<12;l++)
	{
	  x4a[l][k] += x1oda[k];
	}
    }

  //***if x1a....x4a are printed at this point, they should be similar to x1oda....x4oda.
  
  x1a[2] += rise;
  x2a[2] += rise;
  x3a[2] += rise;
  for(int l=3;l<12;l++)
    {
      x4a[l][2] += rise;
    }
  
  //twist and rise  
  zaxis[0] = 0; zaxis[1] = 0; zaxis[2] = 1;
  rotatepointalongaxisbyangle(&x1b[0],&x1a[0],-1.0*twist,&zaxis[0]);
  rotatepointalongaxisbyangle(&x2b[0],&x2a[0],-1.0*twist,&zaxis[0]);
  rotatepointalongaxisbyangle(&x3b[0],&x3a[0],-1.0*twist,&zaxis[0]);
  for(int l=3;l<12;l++)
    {
      rotatepointalongaxisbyangle(&x4b[l][0],&x4a[l][0],-1.0*twist,&zaxis[0]);
    }
  
  //unrotate and untranslate
  for(int k=0;k<3;k++)
    {
      x1b[k] -= x1oda[k];
      x2b[k] -= x1oda[k];
      x3b[k] -= x1oda[k];
      for(int l=3;l<12;l++)
	x4b[l][k] -= x1oda[k];
    }
      
  rotatepointalongaxisbyangle(&x1a[0],&x1b[0],-theta13,&cross2[0]);
  rotatepointalongaxisbyangle(&x2a[0],&x2b[0],-theta13,&cross2[0]);
  rotatepointalongaxisbyangle(&x3a[0],&x3b[0],-theta13,&cross2[0]);
  for(int l=3;l<12;l++)
    rotatepointalongaxisbyangle(&x4a[l][0],&x4b[l][0],-theta13,&cross2[0]);

  rotatepointalongaxisbyangle(&x1b[0],&x1a[0],-theta12,&cross[0]);
  rotatepointalongaxisbyangle(&x2b[0],&x2a[0],-theta12,&cross[0]);
  rotatepointalongaxisbyangle(&x3b[0],&x3a[0],-theta12,&cross[0]);
  for(int l=3;l<12;l++)
    rotatepointalongaxisbyangle(&x4b[l][0],&x4a[l][0],-theta12,&cross[0]);
    
  for(int k=0;k<3;k++)
    {
      xvirt[ivirt][0][k] = x1b[k] + xtrans[k];
      xvirt[ivirt][1][k] = x2b[k] + xtrans[k];
      xvirt[ivirt][2][k] = x3b[k] + xtrans[k];
      for(int l=3;l<12;l++)
	xvirt[ivirt][l][k] = x4b[l][k] + xtrans[k];
    }
}

void FixUCGStateTrans3_3_1::computemoldesum()
{
  int nmolp1=nmol+1;
  double fforce[2],bond_fforce[2]; //value not used
  double factor_lj, factor_coul;
  factor_lj = factor_coul = 1.0; //*//*********************************WRONG************************//*//
  
  
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
  
  //variables related to fil ID
  char var[]="fil";
  int flag=-10;
  int atomfil=-100;
  atomfil=atom->find_custom(var,flag);
  int *afil=atom->ivector[atomfil]; //the variable afil can be used to selectively prevent interactions between beads based on differences of afil for two beads
  int i_afil_temp, j_afil_temp;

  //Pair//
  //similar to pair_lj_cut.cpp

  {
    for(int ii=0;ii<inum;ii++)
      {
	int i = ilist[ii];
	if(!(mask[i] & groupbit)) continue; //similar to fix_bond_swap.cpp, Aram's code.
	int imol = molecule[i];
	int ispecies=mol_species[imol];
	if(trans_flag[imol]==0) continue;
	int itype = type[i];
	int newitype = getnewatomtype(tag[i],imol,itype,ispecies,0);
	i_afil_temp = afil[i];
       
	jlist = firstneigh[i];
	jnum = numneigh[i];
	for(int jj=0;jj<jnum;jj++)
	  {
	    int j = jlist[jj];
	    factor_lj = special_lj[sbmask(j)]; //Is set to 0 if ignoring pair energies between bonded atoms
	    j &= NEIGHMASK;
	    if(!(mask[j] & groupbit)) continue; //Required here too, since we want j to be in the group
	    int jmol = molecule[j];
	    int jspecies=mol_species[jmol];
	    int jtype = type[j];
	    int newjtype;

	    if(jtype>ntypereal) continue; //added to avoid inclusion of virtual particles in energy calculations
	    if(imol==jmol && tag[i]>tag[j]) continue; //skip j>i to avoid double counting
	    if(trans_flag[jmol]==0 || imol != jmol) newjtype = jtype; //keep old jtype if i,j are not same molecules or if they are but jmol is not selected for transition.
	    else 
	      {
		newjtype = getnewatomtype(tag[j],jmol,jtype,jspecies,0);//I.e. new jtype if (jmol==imol and trans_flag[jmol] is set). Isnt this the same as if(trans_flag[jmol]), coz when imol=jmol new jtype
		j_afil_temp = afil[j];
	      }

	    double dx = x[i][0]-x[j][0], dy = x[i][1]-x[j][1], dz = x[i][2]-x[j][2];
	    double rsq = dx*dx+dy*dy+dz*dz;
	    if(rsq<cutsq[itype][jtype])
	      {
		//if(imol==4)
		//printf("imol:%d jmol:%d i:%d j:%d itype:%d jtype:%d newitype:%d newjtype:%d\n",imol,jmol,tag[i],tag[j],itype,jtype,newitype,newjtype);
		double energy_old = force->pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fforce[0]);

		if(mol_filamentID_poly[imol]!=0)
                  afil[i]=abs(mol_filamentID_poly[imol]);
                if(mol_filamentID_poly[imol]!=0 && (imol==jmol))
                  afil[j]=abs(mol_filamentID_poly[jmol]);

		double energy_new = force->pair->single(i,j,newitype,newjtype,rsq,factor_coul,factor_lj,fforce[1]);
		//if(imol==jmol)
		mol_desum[imol] += energy_new-energy_old;
		//else
		//mol_desum[imol] += 0.5*(energy_new-energy_old);
		
                if(mol_filamentID_poly[imol]!=0)
                  afil[i]=i_afil_temp;
                if(mol_filamentID_poly[imol]!=0 && (imol==jmol))
                  afil[j]=j_afil_temp;
		
	      } 
	  }
          
      }
  }

  //Bond//
  //similar to bond_harmonic.cpp//
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
      }//n loop
  }
  //Sum all the energy differences to get the total energy difference for each mol
  MPI_Reduce(&mol_desum[0],&mol_desum_global[0],nmolp1,MPI_DOUBLE,MPI_SUM,0,world);//DO ALL PROCS NEED THIS, OR IS me=0 ENOUGH?
}

void FixUCGStateTrans3_3_1::addforcetomonomers_virtualsitesarenotparticles()
{
  //Neighbor list variables//
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int jnum;
  int *jlist;

  //per atom variables//
  double **x = atom->x;
  double **f = atom->f;
  double *special_lj = force->special_lj;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  const bigint natoms = atom->natoms;

  //local variables
  double dx[3],drsq,fforce,factor_lj;
    
  //add force on state=0 molecules
  for(int ii=0;ii<inum;ii++)
    {
      int i = ilist[ii];
      if(!(mask[i] & groupbit)) continue; //similar to fix_bond_swap.cpp, Aram's code.
      if((tag[i]-1)%nspecies_beads[0]!=1) continue; //only consider one bead of the barbed end
      int imol = molecule[i];
      int endflag=false;
      for(int ifil=1;ifil<=nfilaments;ifil++)
        {
	  if(imol==filament_ends[ifil][0] || imol==filament_ends[ifil][1])
            {
	      int ivirt = 2*(ifil-1);
	      if(imol==filament_ends[ifil][1]) ivirt += 1;
                
	      jlist = firstneigh[i];
	      jnum = numneigh[i];
	      for(int jj=0;jj<jnum;jj++)
                {
		  int j = jlist[jj];
		  factor_lj = special_lj[sbmask(j)]; //Is set to 0 if ignoring pair energies between bonded atoms
		  j &= NEIGHMASK;
		  if(!(mask[j] & groupbit)) continue; //Required here too, since we want j to be in the group
		  int jmol = molecule[j];
		  int jstate=mol_state[jmol];
		  if(jstate==0)
                    {
		      int jd = (tag[j]-1)%nspecies_beads[0];
		      if(jd<4)
                        {
			  for(int k=0;k<3;k++)
			    dx[k] = xvirt[ivirt][jd][k]-x[j][k];
                            
			  drsq = dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2];
			  if(drsq<40000)
                            {
			      //Inverted Gaussian (LAMMPS gauss/cut) sigma=210/3 \AA, H/sigmasqrt(2pi) = 3 Kcal/mol, r_mh=0 ==> U = 3*exp(-r^2/9800)
                                
			      fforce = 0.0006122*exp(-drsq/9800.0);
			      for(int k=0;k<3;k++)
				f[j][k] += fforce*dx[k];
                                
                            }
                        }
                    }
                }
            }
        }
    }
     
}

void FixUCGStateTrans3_3_1::updatevirtualparticletype()
{
  if(!polymerizationflag) return;

  //Neighbor list variables//
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int jnum;
  int *jlist;

  //per atom variables//
  double **x = atom->x;
  imageint *image = atom->image;
  double **f = atom->f;
  double *special_lj = force->special_lj;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;
  int nmolp1=nmol+1;
    
  //Virtual particle types update//
  int ivirt,id,imol;
  int migrateatomsflag1=0;
    
  //--Add force to actin monomers--//

  for(int ifil=1;ifil<=nfilaments;ifil++)
    {
      imol = filament_ends[ifil][0];
      ivirt = 2*(ifil-1)+0;
      findvirtualsite(ivirt,imol,false,1);

      imol = filament_ends[ifil][1];
      ivirt = 2*(ifil-1)+1;
      findvirtualsite(ivirt,imol,true,1);
    } 
    
  for(int i=0;i<nlocal;i++)
    {
      if(!(mask[i] & groupbit)) continue; // i belongs to the group
      imol = molecule[i];
      if(imol<=nmolreal) continue; //imol is not a virtual particle
      
      if(imol<=nmolreal+2*nfilaments)
        {
	  if(type[i]<=ntypereal+12)
            { type[i]+=12; migrateatomsflag1=1;}
	  id = (tag[i]-mol_endid[imol-1]-1)%12;
	  ivirt = (imol-nmolreal-1);
	  for (int k=0;k<3;k++)
	    x[i][k] = xvirt[ivirt][id][k];
	  domain->remap(x[i],image[i]);
        }
    }
  MPI_Allreduce(&migrateatomsflag1,&migrateatomsflag,1,MPI_INT,MPI_MAX,world);

  if(migrateatomsflag!=0)
    next_reneighbor = update->ntimestep; //Trigger reneighboring at the next timestep
}

void FixUCGStateTrans3_3_1::pre_exchange()
{
  int nlocal = atom->nlocal;
  tagint *molecule = atom->molecule;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  double **x = atom->x;
  int nmolp1=nmol+1;
  imageint *image=atom->image;
  bigint natoms = atom->natoms;

  if(migrateatomsflag!=0)
    {        
      for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);
      if (domain->triclinic) domain->x2lamda(atom->nlocal);
      Irregular *irregular = new Irregular(lmp);
      if(irregular->migrate_check()) 
	irregular->migrate_atoms();
      delete irregular;
      if (domain->triclinic) domain->lamda2x(atom->nlocal);

      // check if any atoms were lost
      bigint nblocal = atom->nlocal;
      MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
      if (natoms != atom->natoms && comm->me == 0) {
	char str[128];
	sprintf(str,"Lost atoms via displace_atoms: original " BIGINT_FORMAT
		" current " BIGINT_FORMAT,atom->natoms,natoms);
	error->warning(FLERR,str);
      }

      migrateatomsflag=0;
    }
  //printf("Exiting preexchange  %d\n",me);
}

void FixUCGStateTrans3_3_1::initial_integrate(int vflag)
{    
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

  /*for(int i=0;i<nlocal;i++)
    {
      int imol=molecule[i];
      if(mol_endid[imol]<0)
        printf("coord: %d %d %f %f %f %d\n",tag[i],imol,x[i][0],x[i][1],x[i][2],mol_endid[imol]);
	}*/

  updatevirtualparticletype();
  //printf("Exiting initialintegrate %d\n",me);
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
  factor_lj = factor_coul = 1.0; //*//*********************************WRONG************************//*//

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

  //////needed for the extra per-atom quantity defined. has to be updated after polym/depolym
  char var[]="fil";        
  int flag=-10;                                                                                                                                         
  int atomfil=-100;                                                                                                                                            
  atomfil=atom->find_custom(var,flag);                                                                                                                          
  int *afil=atom->ivector[atomfil];      


  // reallocate work arrays if necessary
  if (atom->nmax > nmax) {
    memory->destroy(atom_state);
    nmax = atom->nmax;
    memory->create(atom_state,nmax,"fix:atomstate");
    vector_atom = atom_state;
  }
  for(int i=0;i<nlocal;i++) {int imol=molecule[i]; atom_state[i]=mol_state[imol];}
  
  //0--Reset--//
  //1--Loop through all molecules and set trans_flag, 0th process only--//
  //2--If change_flag is true--//
  //2a-calculate energy difference-//
  //2b-Meteropolis-Hasting acceptance criterion, 0th process only-//
  //2c-If accept_flag is true, go ahead and update atom types and bond types-//

  //0--Reset--//
  reset_accumulators(); 
 
  //0b--Find dihedral angle for all molecules--//
  for(int i=0;i<nlocal;i++)
    {
      if(!(mask[i] & groupbit)) continue; // i belongs to the group
      int imol = molecule[i];
      int id = (tag[i]-1)%nspecies_beads[0];
      if(mol_species[imol]!=0)
	continue;
      if(id<12)
        {
	  double unwrap[3];
	  domain->unmap(x[i],image[i],unwrap);
	  xdihed[imol][id][0]=unwrap[0];xdihed[imol][id][1]=unwrap[1];xdihed[imol][id][2]=unwrap[2];
	  //////xdihed[imol][id][0] = x[i][0]; xdihed[imol][id][1] = x[i][1]; xdihed[imol][id][2] = x[i][2];

	  //if((long)update->ntimestep==15014933)
	  //printf("coord: %d %d %f %f %f %f %f %f %u\n",tag[i],id,x[i][0],x[i][1],x[i][2],unwrap[0],unwrap[1],unwrap[2],mol_endid[imol]);
	}
	  
    }
  MPI_Allreduce(MPI_IN_PLACE,&xdihed[0][0][0],nmolp1*36,MPI_DOUBLE,MPI_SUM,world);
  if(me==0)
    {
      for(int imol=1;imol<=nmolreal;imol++)
        {
	  if(mol_species[imol]!=0)
            {mol_dihed[imol]=0;continue;}
	  double x1[3],x2[3],x3[3],x4[3];
	  for(int k=0;k<3;k++)
            {
	      x2[k] = xdihed[imol][0][k];
	      x1[k] = xdihed[imol][1][k];
	      x3[k] = xdihed[imol][2][k];
	      x4[k] = xdihed[imol][3][k];
            }
	  double dihedral = 0; 
	  dihedral = computedihedral(x1,x2,x3,x4);
	  mol_dihed[imol] = acos(dihedral)*180/3.14159-180;
        }
    }
   
  if (polymerizationflag)
    findcoordsforpolym((long)update->ntimestep);

  //**The following function assumes that virtual sites are not particles. This does not obey Newton's 3rd law, since there is force on monomers, but no reaction force on the virtual site. Delete if virtual sites are made particles later.**//

  //      addforcetomonomers_virtualsitesarenotparticles(); NOT USING ANYMORE SINCE IT VIOLATES NEWTWON'S THIRD LAW, INSTEAD, INITIAL_INTEGRATE TAKES CARE OF ASSIGNING VIRTUAL PARTICLE POSITIONS, AND FORCEFIELD TAKES CARE OF INTERACTIONS WITH MONOMERS.
      
  //1--Loop through all molecules and set trans_flag, 0th process only--//
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

		if(polymerizationflag && (mol_species[imol]==0) && (j<nstates[0]) && !allowtransition(imol,istate,j)) 
		  {
		    continue; // skips transitions from monomer to f-actin for monomers that are far away from the two ends
		  }

		if(reax_dir[istate][j]<0)
		  rate = rate*(0.5+0.5*tanh(eta[istate][j]*(mol_dihed[imol]-phi0[istate][j])));
		else
		  rate = rate*(0.5-0.5*tanh(eta[istate][j]*(mol_dihed[imol]-phi0[istate][j])));
	        
		if (eta[istate][j]==0)
		  rate=rates[istate][j];
		
		if ((mol_filamentID_poly[imol]<0) || (mol_filamentID_depoly[imol]<0))
		  rate*=scaling[istate][j]; //to account for varying  poly/depoly rate at pointed end for g-actin-ATP monomer
		
		rsum += rate;

		if(rand<rsum)
		  {
		    trans_flag[imol] = j-istate; // add this value to current mol_state to get new state
		    change_flag = 1; //Do not have "break" here. Allow selecting multiple molecules
#ifdef MYDEBUG1
		    printf("UCG selected mol %d with state %d direction %d at step %ld with rate %f %f and rand %f\n",imol,istate,trans_flag[imol],(long)update->ntimestep,rsum,rate,rand);
#endif
		    break;		    
		  }
		mol_filamentID_poly[imol]*=trans_flag[imol];
		mol_filamentID_depoly[imol]*=trans_flag[imol];
	      }
	  }
      }
    MPI_Bcast(&change_flag,1,MPI_UNSIGNED_SHORT,0,world);
    MPI_Bcast(&mol_filamentID_poly[0],nmol+1,MPI_INT,0,world);
    MPI_Bcast(&mol_filamentID_depoly[0],nmol+1,MPI_INT,0,world);   
  }

#ifdef MYDEBUG1
  //////to delete later
  if(me==0)
    {
      std::ofstream write("mol_transflag");
      write<<"trans_flag"<<' '<<"mol_filamentID_poly"<<' '<<"mol_filamentID_depoly"<<"\n";
      for (int imol=1;imol<=nmol;imol++)
	write<<imol<<' '<<trans_flag[imol]<<' '<<mol_filamentID_poly[imol]<<' '<<mol_filamentID_depoly[imol]<<"\n";
      write.close();
    }
  //////to delete later
#endif
   
  //2--If change_flag is true--//
  if(change_flag==1)
    {
      MPI_Bcast(&trans_flag[0],nmolp1,MPI_SHORT,0,world);
      //2a-calculate energy difference-//

      computemoldesum();
      
      //2b-Meteropolis-Hasting acceptance criterion, 0th process only-/
      //Metropolis-Hastings acceptance criterion
      if(me==0)
        {
	  accept_flag=0;
	  for(int imol=1;imol<=nmolreal;imol++)
            {
	      int istate=mol_state[imol];
	     
	      if(trans_flag[imol]==0) continue;
	      int k;
	      double detailedbalancefactor=1.0;
	      //forward reaction, k=0, reverse reaction, k=1.
	      if(reax_dir[istate][istate+trans_flag[imol]]<0)//forward
                { 
		  k=0;  
		  detailedbalancefactor=(rates[istate+trans_flag[imol]][istate])/(rates[istate][istate+trans_flag[imol]]);
		  detailedbalancefactor *= ( (0.5-0.5*tanh(eta[istate+trans_flag[imol]][istate]*(mol_dihed[imol]-phi0[istate+trans_flag[imol]][istate]))) )/( (0.5+0.5*tanh(eta[istate][istate+trans_flag[imol]]*(mol_dihed[imol]-phi0[istate][istate+trans_flag[imol]]))) );
                }
     
	      else //reverse 
                { 
		  k=1;  
		  detailedbalancefactor=(rates[istate+trans_flag[imol]][istate])/(rates[istate][istate+trans_flag[imol]]);
		  detailedbalancefactor *= ( (0.5+0.5*tanh(eta[istate+trans_flag[imol]][istate]*(mol_dihed[imol]-phi0[istate+trans_flag[imol]][istate]))) )/( (0.5-0.5*tanh(eta[istate][istate+trans_flag[imol]]*(mol_dihed[imol]-phi0[istate][istate+trans_flag[imol]]))) );
                }
	      
	      rand = randomp->uniform();	
	      mhterm = MIN(detailedbalancefactor*exp(-Beta*mol_desum_global[imol] + mhcorr[istate][istate+trans_flag[imol]]), 1.0);

#ifdef MYDEBUG1
	      if(me==0)
		printf("Info: %f %f %f %f %f %f\n",rand,mhterm,detailedbalancefactor,Beta,mol_desum_global[imol],-Beta*mol_desum_global[imol] + mhcorr[istate][istate+trans_flag[imol]]);
#endif
	     
	      if(rand<mhterm || mhterm==1.0)
                {
		  mol_accept[imol] = 1;
		  mol_state[imol] += trans_flag[imol]; //update//
		  accept_flag = 1;
                }
#ifdef MYDEBUG
	      else
		printf("U m %d s %d -> %d f s %ld dU %f dihed %f beta %f p %d d %d\n",imol,mol_state[imol],mol_state[imol]+trans_flag[imol],(long)update->ntimestep,mol_desum_global[imol],mol_dihed[imol],Beta,mol_filamentID_poly[imol],mol_filamentID_depoly[imol]);
#endif
            }
        }//me==0 if
      MPI_Bcast(&accept_flag,1,MPI_UNSIGNED_SHORT,0,world);
      
     

      //2c-If accept_flag is true, go ahead and update atom types and bond types-//
      if(accept_flag==1)
        {
	  next_reneighbor = update->ntimestep; //Trigger reneighboring at the next timestep. This is required since atom types are updated here.//
	  MPI_Bcast(&mol_accept[0],nmol+1,MPI_UNSIGNED_SHORT,0,world); //communicate mol_accept
	  MPI_Bcast(&mol_state[0],nmol+1,MPI_SHORT,0,world); //communicate mol_state

	  set_filamentID(1,(long)update->ntimestep,Beta);

	 
#ifdef MYDEBUG1	  
	  //////to delete later
	  if(me==0)
            {
	      std::ofstream write("mol_state_accept");
	      write<<"molID"<<' '<<"mol_state"<<' '<<"mol_accept"<<"\n";
	      for(int imol=1;imol<nmol+1;imol++)
		write<<imol<<' '<<mol_state[imol]<<' '<<mol_accept[imol]<<"\n";
	      write.close();
            }
	  //////to delete later
#endif
	  
	  //Atom types update//
	  for(int i=0;i<nlocal;i++)
            {
	      if(!(mask[i] & groupbit)) continue; // i belongs to the group
	      int imol = molecule[i];
	      atom_state[i] = mol_state[imol];
	      if(mol_accept[imol]==0) continue;
	      int itype = type[i];
	      int ispecies=mol_species[imol];
	      int newitype=getnewatomtype(tag[i],imol,itype,ispecies,1);
	      type[i]=newitype;
#ifdef MYDEBUG1
	      //////printf("UCG changed %d atom's type from %d to %d at step %d\n",tag[i],itype,newitype,(long)update->ntimestep);//DEBUG
#endif

            }
	  
	  //Bond types update//
	  //similar to fix_bond_swap.cpp
	  //not using bondlist here,because neigh_bond.cpp creates bondlist using bond_atom and bond_type. Hence, need to overwrite/update bond_type and trigger build_topology.
	  for(int i=0;i<nlocal;i++)
            {
	      if(!(mask[i] & groupbit)) continue; //
	      int imol = molecule[i];
	      int ispecies=mol_species[imol];
	      if(mol_accept[imol]==0) continue; //CHECK WHAT HAPPENS WITH INEXT
	      for(int ibond=0;ibond<num_bond[i];ibond++)
                {
		  int inext = atom->map(bond_atom[i][ibond]);
		  //if(inext >= nlocal || inext < 0) continue; //Not using this because bond list is unique. i.e. if 1-2 exists, 2-1 does not.
		  if(molecule[inext] != imol) error->all(FLERR,"Inter-molecule bond in fix ucg_state_trans2");//DEBUG
		  int ibondtype = bond_type[i][ibond];
		  int newibondtype = ibondtype + trans_flag[imol]*bondtype_offset[ispecies];
		  bond_type[i][ibond] = newibondtype; //update//
#ifdef MYDEBUG1
		  ////// printf("UCG changed %d-%d bond's type from %d to %d at step %d\n",tag[i],inext+1,ibondtype,newibondtype,(long)update->ntimestep);//DEBUG
#endif
                }
            } 
	  neighbor->build_topology();  //Force rebuiliding of topology since bond parameters have changed. This is required here, after updating bond_type.//

        }//accept_flag if
    }//change_flag if
 

  if(nprocs > 1) comm->forward_comm_fix(this);
  //printf("Exiting postforce %d\n",me);
 
}

/* ---------------------------------------------------------------------- */
void FixUCGStateTrans3_3_1::end_of_step()
{
  //per-atom variables
  double **x = atom->x;
  imageint *image = atom->image;
  double **f = atom->f;
  double *special_lj = force->special_lj;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;
  int nmolp1=nmol+1;


  //////needed for the extra per-atom quantity defined. has to be updated after polym/depolym
  char var[]="fil";        
  int flag=-10;                                                                                                                                         
  int atomfil=-100;                                                                                                                                            
  atomfil=atom->find_custom(var,flag);                                                                                                                          
  int *afil=atom->ivector[atomfil];      

  t_current = temperature->compute_scalar();

  //std::ofstream write("Interaction_factor");
 
  for (int i = 0; i<nlocal; i++)
    {
      int imol = molecule[i]; //atom_filamentID[i] = mol_filamentID[imol];
 
      if(imol>nmolreal) continue; //not update the virtual particles. should be zero always
	         
      /*if(interaction_factor[imol]<1)
	{
	  interaction_factor[imol]=((long)update->ntimestep-timer[imol])/(100000);	  
	  }*/

      if(mol_filamentID[imol]>0)
	{
	  //interaction_factor[imol]=1;
	  //timer[imol]=0;			    
	  afil[i]=abs(mol_filamentID[imol]);
	}
      /*else if(mol_filamentID_depoly[imol]!=0)
	{	
	  //interaction_factor[imol]=0;
	  //timer[imol]=(long)update->ntimestep;
	  afil[i]=-1*imol; 
	  }*/
      else
	{
	  // afil[i]=-imol*floor(interaction_factor[imol]);
	  afil[i]=-1*imol;
	}

      //write<<"imol: "<<imol<<' '<<"interaction_factor: "<<interaction_factor[imol]<<' '<<"index: "<<tag[i]<<' '<<"afil: "<<afil[i]<<' '<<"mol_filamentID: "<<mol_filamentID[imol]<<' '<<"timer: "<<timer[imol]<<' '<<"current step: "<<(long)update->ntimestep<<"\n";
      
    }
  //write.close();
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

      for(int j=0;j<12;j++)
	for(int k=0;k<3;k++)
	  xdihed[imol][j][k] = 0.0;

      xpolym[imol][0] = xpolym[imol][1] = xpolym[imol][2] = 0;
      xpolymtype[imol]=0;
      mol_filamentID_depoly[imol]=0;
      mol_filamentID_poly[imol]=0;
    }

  for (int i = 0; i < (nfilaments + 1); i++)
    {
      filament_endspolymflag[i][0] = filament_endspolymflag[i][1] = 0;
      filament_endsdepolymflag[i][0] = filament_endsdepolymflag[i][1] = 0;
    }
}
/*------------------------------------------------------------------------*/

void FixUCGStateTrans3_3_1::set_filamentID(int update,long int step,double beta)
{
  int nmolp1 = nmol + 1;
  int nmolrealp1=nmolreal+1;
  int dnfid=0, nfid=-1;
  int imolstate;
  int mfid;
  
  tagint *molecule = atom->molecule;

  if (!update && polymerizationflag)
    {
      //this section has to change when the number of filaments increase in the system  
      for (int imol = 0; imol < nmolp1; imol++)
        {
	  mol_filamentID_poly[imol] = 0;
	  mol_filamentID_depoly[imol] = 0;
	}      
    }
  else
    {
      int Loc;
      int oldmol;

      for (int imol = 1; imol < nmolrealp1; imol++)
        {
	  if(polymerizationflag)
	    {
	      if (mol_accept[imol] == 1 && mol_species[imol]==0)
		{

		  if ((mol_filamentID_poly[imol]!= 0))  //polymerization if 
		    {

		      int filid=abs(mol_filamentID_poly[imol]);
		      Loc = mol_filamentID_poly[imol] > 0 ? 0 : 1;
		  
		      if( ((mol_filamentID_depoly[filament_ends[filid][Loc]]!=0) && (mol_accept[filament_ends[filid][Loc]]==1)) || (filament_endspolymflag[filid][Loc]!=0) || (filament_endsdepolymflag[filid][Loc]!=0) )
			{
			  mol_accept[imol]=0;mol_state[imol]-=trans_flag[imol];
			  continue;
			}
		  
		      filament_endspolymflag[filid][Loc]=1;
		      mol_filamentID[imol] = abs(mol_filamentID_poly[imol]);
		      oldmol = filament_ends[abs(mol_filamentID_poly[imol])][Loc];
		      filament_ends[abs(mol_filamentID_poly[imol])][Loc] = imol;

		      if (Loc == 0)
			{
			  mol_filamentlistforward[imol] = oldmol;
			  mol_filamentlistbackward[oldmol] = imol;
			  mol_filamentlistbackward[imol] = 0;
			}
		      else
			{
			  mol_filamentlistforward[oldmol] = imol;
			  mol_filamentlistforward[imol] = 0;
			  mol_filamentlistbackward[imol] = oldmol;
			}
		    }

		  else if (mol_filamentID_depoly[imol] != 0) //depolymerization elseif 
		    {
		      mol_filamentID[imol] = 0;

		      Loc = mol_filamentID_depoly[imol] > 0 ? 0 : 1;
		      oldmol = filament_ends[abs(mol_filamentID_depoly[imol])][Loc];
				  
		      if(oldmol!=imol)
			error->all(FLERR, "molecule undergoing depolymerization is not an end of a filament. possible error in source code of fix ucg_state_trans3 command");

		      filament_endsdepolymflag[abs(mol_filamentID_depoly[imol])][Loc]=1;
		      if (Loc == 0)
			{
			  filament_ends[abs(mol_filamentID_depoly[imol])][Loc] = mol_filamentlistforward[oldmol];
			  mol_filamentlistforward[oldmol] = 0;
			  mol_filamentlistbackward[filament_ends[abs(mol_filamentID_depoly[imol])][Loc]] = 0;
			}
		      else
			{
			  filament_ends[abs(mol_filamentID_depoly[imol])][Loc] = mol_filamentlistbackward[oldmol];
			  mol_filamentlistforward[filament_ends[abs(mol_filamentID_depoly[imol])][Loc]] = 0;
			  mol_filamentlistbackward[oldmol] = 0;
			}
		    }

		  else   //other transitions else
		    {
		 
		    }
		}
	    }
	  if(mol_accept[imol]==1 && me==0)
	    {
	      printf("U m %d s %d -> %d s s %ld dU %f dihed %f beta %f p %d d %d\n",imol,mol_state[imol]-trans_flag[imol],mol_state[imol],step,mol_desum_global[imol],mol_dihed[imol],beta,mol_filamentID_poly[imol],mol_filamentID_depoly[imol]);
	    }
	  
        }
    }

  
  if(me==0)
    {
       printf("Total filament %d\n",nfilaments);
       printf("FilamentID \t BarbedEndMolID \t PointedEndMolID\n");
       for(int i=0;i<=nfilaments;i++)
	 printf("%d \t %d \t %d \n",i,filament_ends[i][0],filament_ends[i][1]);
      
#ifdef MYDEBUG1
      //////to delete later
      std::ofstream write("mol_linkedlist");
      // write<<"Timestep:"<<update->ntimestep<<"\n";
      write<<"molID"<<' '<<"ForwardList"<<' '<<"BackwardList"<<"FilamentID"<<"\n";
      for(int imol=1;imol<nmolp1;imol++)
	write<<imol<<' '<<mol_filamentlistforward[imol]<<' '<<mol_filamentlistbackward[imol]<<' '<<mol_filamentID[imol]<<"\n";
      write.close();
      //////to delete later
#endif

    }	

  //printf("Exiting setfilamentID %d\n",me);
}

/*----------------------------------------------------------------------------------*/
int FixUCGStateTrans3_3_1::getnewatomtype(int id, int imol, int itype, int ispecies, int update)
{
  double **x = atom->x;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  const bigint natoms = atom->natoms;
  int nmolp1 = nmol + 1;

  /*works only for g-actin monomer state as zero*/
  if((polymerizationflag) && (mol_species[imol]==0) && (mol_filamentID_poly[imol] != 0))
    {
      int neighmol=0, neightype, neighid, neighmolpos;
      int imolbasetype, imolpostype, newitype, idloc;
      if (mol_filamentID_poly[imol] > 0)
        {
	  neighmol = update > 0 ? mol_filamentlistforward[imol] : filament_ends[abs(mol_filamentID_poly[imol])][0];
	  neightype = xpolymtype[neighmol];
	
	  //neighmolpos = ceil((neightype - (mol_state[neighmol] * atomtype_offset[0])) / nspecies_beads[0]); //commented as per correction from Vil4Site.cpp
	  neighmolpos = ceil((neightype - ((mol_state[neighmol]-trans_flag[neighmol]*mol_accept[neighmol]) * atomtype_offset[0])) / nspecies_beads[0]); 
	  
	  //imolpos = neighmolpos - 1;
	  imolbasetype = (mol_state[imol]+!update*trans_flag[imol]) * atomtype_offset[0] + atomtype_offset[0];
	  imolpostype = imolbasetype - ((atomtype_offset[0] / nspecies_beads[0]) - (neighmolpos - 1))*nspecies_beads[0];
	  idloc = mol_endid[imol] - id + 1;
	  idloc = nspecies_beads[0]-idloc + 1;
	  newitype = imolpostype - nspecies_beads[0] + idloc;
	  newitype = newitype + !(neighmolpos - 1)*atomtype_offset[0];
	  
	  if(newitype<0)
	    printf("polym: imol:%d neighmol:%d itype:%d neightype:%d newitype:%d endid:%d %d\n",imol,neighmol,itype,neightype,newitype,mol_endid[imol],update);

	  return newitype;
        }
      else
        {
	  neighmol = update > 0 ? mol_filamentlistbackward[imol] : filament_ends[abs(mol_filamentID_poly[imol])][1];
	  neightype = xpolymtype[neighmol];

          //neighmolpos = ceil((neightype - (mol_state[neighmol] * atomtype_offset[0])) / nspecies_beads[0]);
          neighmolpos = ceil((neightype - ((mol_state[neighmol]-trans_flag[neighmol]*mol_accept[neighmol]) * atomtype_offset[0])) / nspecies_beads[0]);

	  imolbasetype = (mol_state[imol]+!update*trans_flag[imol]) * atomtype_offset[0] + atomtype_offset[0];
	  imolpostype = imolbasetype - ((atomtype_offset[0] / nspecies_beads[0]) - (neighmolpos + 1))*nspecies_beads[0];
	  idloc = mol_endid[imol] - id + 1;
	  idloc = nspecies_beads[0] - idloc + 1;
	  newitype = imolpostype - nspecies_beads[0] + idloc;
	  newitype = newitype - !(neighmolpos - (atomtype_offset[0] / nspecies_beads[0]))*atomtype_offset[0];
	  
	  if(newitype<0)
	    printf("polym: imol:%d neighmol:%d itype:%d neightype:%d newitype:%d endid:%d %d\n",imol,neighmol,itype,neightype,newitype,mol_endid[imol],update);

	  return newitype;
        }
    }
  else
    return(itype+trans_flag[imol]*atomtype_offset[ispecies]);
} 

/*-------------------------------------------------------------------------------*/
int FixUCGStateTrans3_3_1::allowtransition(int imol, int istate, int j)
{
  //return 1 to allow , 0 to not allow. function not general for all components.
  //approximation: monomer g-actin is only state 0 for species 0
  
  int returnval=0;
  int xflag=0, yflag=0, zflag=0; 
  int virtmol1=0;//stores the virtual particle ID corres. for each filament to check polymerization
  int virtmol2=0;

  if((istate!=0) && (j!=0))
    { //not a polymerization or depolymerization rxn.
      returnval=1;
    }

  if ((istate!=0) && (j==0)) //depolymerization rxn
    {
      int filid = mol_filamentID[imol];

      if ((imol == filament_ends[filid][0]))
        {
	  returnval = 1;
	  mol_filamentID_depoly[imol] = filid; //positive for depolymerization from barbed end

        }
      else if ((imol == filament_ends[filid][1]))
        {
	  returnval = 1;
	  mol_filamentID_depoly[imol] = -filid; //negative for depolymerization from pointed end
        }
      else
        {
	  returnval = 0;
        }      
    }
  
  if ((istate==0)&&(j!=0)) //polymerization rxn
    {
      for (int i = 1; i < (nfilaments + 1); i++)
        {
	  virtmol1=nmolreal+2*i-1;
	  virtmol2=virtmol1+1;

	  if ((abs(xpolym[imol][0] - xpolym[virtmol1][0]) < 15.0) && (abs(xpolym[imol][1] - xpolym[virtmol1][1]) < 15.0) && (abs(xpolym[imol][2] - xpolym[virtmol1][2]) < 15.0)) 
            {  
	      xflag = yflag = zflag = 1; returnval = 1;
	      mol_filamentID_poly[imol] = i;  //polymerization at barbed end
	      break;
            }
	  else if ((abs(xpolym[imol][0] - xpolym[virtmol2][0]) < 15.0) && (abs(xpolym[imol][1] - xpolym[virtmol2][1]) < 15.0) && (abs(xpolym[imol][2] - xpolym[virtmol2][2]) < 15.0))
            {  
	      xflag = yflag = zflag = 1; returnval = 1;
	      mol_filamentID_poly[imol] = -i;  //polymerization at pointed end
	      break;
            }
	  else
            {
	      returnval = 0;

            }
        }
    }

  return returnval;
}

/*------------------------------------------------------------------------------------------------------*/
void FixUCGStateTrans3_3_1::findcoordsforpolym(long int step)
{
  //approximation: all actins are mentioned before the rest of the components in initialstate.in
  //function not general for all components

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  const bigint natoms = atom->natoms;
  int nmolp1 = nmol + 1;

  for (int i = 0; i<nlocal; i++)
    {
      if (!(mask[i] & groupbit)) continue; // i belongs to the group
      int imol = molecule[i];
      int imolendid = mol_endid[imol]; 
      int id = tag[i];
      //if (mol_species[imol] != 0)
      //continue;
      
      if(imolendid<0)
	printf("tagid: %d endid: %d mol: %d step: %ld\n",tag[i],mol_endid[imol],imol,step);

      if (id==imolendid)
	{
	  xpolym[imol][0] = x[i][0]; xpolym[imol][1] = x[i][1]; xpolym[imol][2] = x[i][2]; xpolymtype[imol] = type[i];
	  
	  //if(step==15014933)
	    //printf("endid:%d %d\n",id,imolendid);

	  //if(imol==5)
	  //printf("imol:%d coords:%f %f %f type:%d step:%ld\n",imol,x[i][0],x[i][1],x[i][2],type[i],step);
        }
    }
  MPI_Allreduce(MPI_IN_PLACE, &xpolym[0][0], (nmolp1 * 3), MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, &xpolymtype[0], nmolp1, MPI_UNSIGNED_SHORT, MPI_SUM, world);

#ifdef MYDEBUG1
  //////to delete later
  if(me==0)
    {
      std::ofstream write("mol_coords");
      write<<"molID"<<' '<<"x"<<' '<<"y"<<' '<<"z"<<' '<<"type"<<"\n";
      for(int imol=1;imol<nmolp1;imol++)
	write<<imol<<' '<<xpolym[imol][0]<<' '<<xpolym[imol][1]<<' '<<xpolym[imol][2]<<' '<<xpolymtype[imol]<<' '<<mol_endid[imol]<<"\n";
      write.close();
    }
  //////to delete later
#endif

}

/*---------------------------------------------------------------------------------------*/

/*
  void FixUCGStateTrans2::reset_accumulators()
  {
  int nmolp1 = nmol+1;
  memset(&trans_flag[0],false,(nmolp1)*sizeof(bool));
  memset(&mol_desum[0],0,(nmolp1)*sizeof(double));
  memset(&mol_accept[0],false,(nmolp1)*sizeof(bool));
  
  #ifdef MYDEBUG
  memset(&mol_esum_oldstate[0],0,(nmolp1)*sizeof(double));
  #endif
  }

*/
