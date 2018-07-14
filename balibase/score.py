from os.path import isfile
import subprocess
import shlex


class BAliBASEError(Exception):
    pass


class BAliBASERuntimeError(BAliBASEError):
    pass


class BAliBASEIOError(BAliBASEError):
    pass


def score(reffile, msffile):
    if isfile(reffile) and isfile(msffile):
        cmd = "./bali_score %s %s" % (reffile, msffile)
        try:
            output = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            output = e.output
        output = output.split("\n")
        if len(output) >= 2:
            result = output[-2].split()
            if result[0] == "auto":
                return float(result[2]), float(result[3])
            else:
                raise BAliBASERuntimeError(output[0])
        else:
            raise BAliBASERuntimeError(output[0])
    else:
        if not isfile(reffile):
            raise BAliBASEIOError('No such reference file: %s' % reffile)
        if not isfile(msffile):
            raise BAliBASEIOError('No such alignment file: %s' % msffile)


__all__ = ["score", "BAliBASEError", "BAliBASERuntimeError", "BAliBASEIOError"]
