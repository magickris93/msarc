from argparse import ArgumentParser, RawDescriptionHelpFormatter
from multiprocessing import cpu_count


def parse(nargs=1, allow_verbose=False, multiprocessing=False):
    from .alignment import CONSISTENCY, REFINE
    from .probA import (TEMPERATURE_DEFAULT, ENDGAP_DEFAULT, ENDGAP_NONE,
                        OPENGAP_DEFAULT, EXTENDGAP_DEFAULT,
                        TYPE_AUTO, TYPE_PROTEIN, TYPE_DNA,
                        MATRIX_DEFAULT, SET_DEFAULT, CUT_DEFAULT)

    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            epilog="\nSupported substitution matricies and sets:\n"
                                   "  blosum:   30, 50, 62, 80, -1 (auto)\n"
                                   "  gonnet:   40, 80, 120, 160, 250, 300, 350, -1 (auto)\n"
                                   "  pam:      20, 60, 120, 350, -1 (auto)\n")
    parser.add_argument("-T", "--temperature", type=float, default=TEMPERATURE_DEFAULT, dest="temperature",
                        help="thermodynamic temperature [default: %(default)s]", metavar="TEMP")
    parser.add_argument("-M", "--multilevel", action="store_true", dest="multilevel",
                        help="use multilevel graph partitioning algorithm")
    parser.add_argument("-C", "--consistency", type=int, default=CONSISTENCY, dest="consistency",
                        help="iterations of consistency transformation [default: %(default)s]", metavar="REPS")
    parser.add_argument("-R", "--refinements", type=int, default=REFINE, dest="refine",
                        help="iterations of horizontal refinement [default: %(default)s]", metavar="REPS")
    parser.add_argument("-c", "--cut", type=float, default=CUT_DEFAULT, dest="cut",
                        help="cut off value for posterior probabilities [default: %(default)s]", metavar="CUT")
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument("-d", "--DNA", action="store_const", const=TYPE_DNA, default=TYPE_AUTO, dest="type",
                            help="input polymers will be treated as nucleic acids")
    type_group.add_argument("-p", "--protein", action="store_const", const=TYPE_PROTEIN, dest="type",
                            help="input polymers will be treated as amino acids")
    parser.add_argument("-g", "--gap-open", type=float, default=OPENGAP_DEFAULT, dest="opengap",
                        help="gap opening penalty [default: from substitution matrix]", metavar="SCORE")
    parser.add_argument("-x", "--gap-extend", type=float, default=EXTENDGAP_DEFAULT, dest="extendgap",
                        help="gap extension penalty [default: from substitution matrix]", metavar="SCORE")
    egap_group = parser.add_mutually_exclusive_group()
    egap_group.add_argument("-e", "--end-gaps", type=float, default=ENDGAP_DEFAULT, dest="endgap",
                            help="terminal gap penalty [default: %(default)s]", metavar="SCORE")
    egap_group.add_argument("--no-end-gaps", action="store_const", const=ENDGAP_NONE, dest="endgap",
                            help="treat terminal gaps as gaps inside alignment")
    parser.add_argument("-m", "--matrix", default=MATRIX_DEFAULT, dest="matrix",
                        choices=["blosum", "gonnet", "pam"],
                        help="substitution matrix series [default: %(default)s]", metavar="MATRIX")
    parser.add_argument("-s", "--set", type=int, default=SET_DEFAULT, dest="set",
                        help="substitution matrix set [default: %(default)s]", metavar="SET")
    parser.add_argument("-w", "--weighted-transformation", action="store_true", dest="weightrans",
                        help="weight sequence pairs in consistency transformation")
    if allow_verbose:
        parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                            help="show progress while aligning.")
    if multiprocessing:
        parser.add_argument("-P", "--max-processes", type=int, default=1, dest="processes",
                            help="maximum number of simultaneous processes [default: %(default)s]", metavar="PROCESSES")
    parser.add_argument("files", metavar="FILE", nargs=nargs,
                        help="a file containing the input polymers in FASTA format.")

    return parser.parse_args(), parser.error
