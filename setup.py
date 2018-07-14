from distutils.core import setup, Extension
from numpy import get_include as np_get_include
try:
    from Cython.Distutils import build_ext
except ImportError:
    cython_source_ext = "c"
    cmdclass = {}
else:
    cython_source_ext = "pyx"
    cmdclass = {"build_ext": build_ext}


setup(name="msarc",
      version="1.1",
      description="Multiple Sequence Alignment by Residue Clustering.",
      ext_modules=[Extension("clusteralign._graph",
                             sources=["clusteralign/_graph.%s" % cython_source_ext],
                             include_dirs=[np_get_include()]),
                   Extension("clusteralign._probA",
                             sources=["clusteralign/_probA.%s" % cython_source_ext,
                                      "probA/matrices.c",
                                      "probA/pf_goto.c"],
                             include_dirs=[np_get_include(),
                                           "probA"]),
                   Extension("clusteralign._ppalign",
                             sources=["clusteralign/_ppalign.%s" % cython_source_ext],
                             include_dirs=[np_get_include()]),
                   Extension("clusteralign._partition",
                             sources=["clusteralign/_partition.%s" % cython_source_ext],
                             include_dirs=[np_get_include()])],
      requires=["numpy",
                "scipy",
                "biopython"],
      cmdclass=cmdclass)
