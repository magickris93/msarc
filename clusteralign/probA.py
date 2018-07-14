from __future__ import print_function
from math import log
from scipy.sparse import csr_matrix
from functools import partial

TEMPERATURE_GONNET = log(10) / 2
TEMPERATURE_DEFAULT = TEMPERATURE_GONNET
OPENGAP_DEFAULT = None
EXTENDGAP_DEFAULT = None
ENDGAP_DEFAULT = 0
ENDGAP_NONE = None
TYPE_AUTO = "auto"
TYPE_PROTEIN = "protein"
TYPE_DNA = "DNA"
MATRIX_DEFAULT = "gonnet"
SET_DEFAULT = -1
CUT_DEFAULT = 0.01


def align(records, type=TYPE_PROTEIN, temperature=TEMPERATURE_DEFAULT,
          matrix=MATRIX_DEFAULT, set=SET_DEFAULT,
          opengap=OPENGAP_DEFAULT, extendgap=EXTENDGAP_DEFAULT,
          endgap=ENDGAP_DEFAULT,
          cut=CUT_DEFAULT, verbose=False, processes=1):
    """Align two sequences using probA."""
    from _probA import align as c_align

    if processes > 1:
        record1, record2, r1, r2 = records
    else:
        record1, record2 = records

    def _lib_type(type):
        types = {TYPE_AUTO: -1,
                 TYPE_PROTEIN: 0,
                 TYPE_DNA: 1}
        try:
            return types[type]
        except KeyError:
            raise ValueError

    if verbose:
        print("  %s with %s ..." % (record1.id, record2.id), end=' ')

    type_i = _lib_type(type)
    pairs, params = c_align(record1, record2, temperature,
                            opengap, extendgap, endgap,
                            type_i, matrix, set, cut)

    if verbose:
        print("done")
        sequence_type, matrix_name, go, ge, tg = params
        print("    matrix: %s" % matrix_name.replace("_", " "))
        print("    gap penalties: %f (open), %f (extend)" % (go, ge), end='')
        if endgap is not None:
            print(", %f (terminal)" % tg, end='')
        print()

    sequence_type = TYPE_PROTEIN if params[0] == 0 else TYPE_DNA
    if processes > 1:
        return csr_matrix(pairs), sequence_type, r1, r2 # return r1 and r2 as well
    else:
        return csr_matrix(pairs), sequence_type


def pairwise(records, verbose=False, **kwargs):
    """Pairwise align all sequences using probA."""
    from numpy import empty

    pairs = empty((len(records), len(records)), dtype=object)
    sequence_type = None
    numprocs=kwargs.get('processes', 1)
    if verbose:
        print("calculating pairwise probabilities ...")

#   MSA wykonywane jest domyslnie algorytmem szeregowym, argument -P wykonuje algorytm rownolegly
    if numprocs > 1:
        import multiprocessing
        palign = partial(align, verbose=verbose, **kwargs)
        pairtuples = sum([[(records[r1], records[r2], r1, r2) for r2 in xrange(r1 + 1, len(records))] for r1 in xrange(len(records))], [])
        pool = multiprocessing.Pool(numprocs)
        results=pool.map(palign, pairtuples)
        pool.close()

        for pair in results:
            result = pair[0]
            _sequence_type = pair[1]
            r1, r2 = pair[2], pair[3]
            pairs[r1, r2] = result
            if sequence_type is None:
                sequence_type = _sequence_type
            elif sequence_type != _sequence_type:
                print()
                print("mixed residue types detected - recalculating as proteins ...")
                print()
                return pairwise(records, verbose=verbose, **dict(kwargs, type=TYPE_PROTEIN))
    else:
        for r1 in xrange(len(records)):
            for r2 in xrange(r1 + 1, len(records)):
                _pairs, _sequence_type = align((records[r1], records[r2]), verbose=verbose, **kwargs)
                pairs[r1, r2] = _pairs

                if sequence_type is None:
                    sequence_type = _sequence_type
                elif sequence_type != _sequence_type:
                    print()
                    print("mixed residue types detected - recalculating as proteins ...")
                    print()
                    return pairwise(records, verbose=verbose, **dict(kwargs, type=TYPE_PROTEIN))

    if verbose:
        print()

    return pairs, sequence_type
