#include "pfgoto.h"

int verbose_flag;    /* Flag set by `-verbose' */

int typ_flag=-1;     /* Flag set by `-DNA'=1  or `-prot'=0, default -1,
			function check_polymer is called */

int PS_flag=1;       /* Flag set by `-PS', default is 1 =>
			postscript output */

int Ogap_flag=0;
int Xgap_flag=0;
int Egap_flag=1;     /* Flag set by `-endgaps', the default value of endgaps
		        is 0, but can be modified by the user. To turn off the
		        endgaps option use -noEg*/

int matrix_flag=0;   /* Flag set by `-score_matrix', requires as argument a
			matrix series */

int dist_flag=-1;    /* Flag set by `-pam', default is -1,
			set to pam distance (gon, pam) or to the observed
			identity (blo) to select one matrix out of a matrix
			series*/
 /*int calcS_flag=0;    Flag set by `-calc_s', default is 0, requires as
			argument a filename containing only the alignment
			encoded as one string of symbols -> calculates the
			score of this alignment*/

/*  the struct option structure has these fields: */
/*     const char *name                                                  */
/*     int has_arg    three legitimate values: no_argument(0)            */
/*  	                                       required_argument(1)      */
/*  	                                       optional_argument(2)      */
/*     int *flag      flag=0; => return val
                      otherwise => return 0; flag points to a variable which
		      is set to val if the option is found    */
/*     int val        value to return, or to load into  the  variable pointed
                      to by flag.*/

struct option long_options[] =
{
  /*  These options set no flag and have no argument */
  {"help", 0, 0, 'h'},
  {"version", 0, 0, 'V'},
  /* These options set a flag & have no argument */
  {"verbose", 0, &verbose_flag, 1},
  {"brief", 0, &verbose_flag, -1},
  {"DNA", 0, &typ_flag, 1},
  {"prot", 0, &typ_flag, 0},
  {"noEg", 0, &Egap_flag, 0},
  {"noPS", 0, &PS_flag, 0},
  /* These options set a flag & have an argument */
  {"pam", 1, &dist_flag, 1},
  {"gapopen", 0, &Ogap_flag,1},
  {"gapextend", 0, &Xgap_flag,1},
  {"endgaps", 1, &Egap_flag,1},
  {"score_matrix", 1, &matrix_flag,1},
  /*{"calc_s", 1, &calcS_flag,1},
   These options don't set a flag.
   We distinguish them by their indices. */
  {"file", 1, 0, 0},
  {0, 0, 0, 0}
};

int option_index = 0;    /* getopt_long stores the option index here. */

/*short options
  o:  required argument
  o:: optional argument

  If the option has a required argument,  it  may be
  written  directly after the option character or as the
  next parameter (ie. separated by whitespace on the command
  line).  If the option has an optional argument, it must be
  written directly after the option character if present.

  T modifies BETA
  N number of alignments generated by the stocastic backtracking
  h help
  V Version */
char *shortopts = "hVT:N:";








