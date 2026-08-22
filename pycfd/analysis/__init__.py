"""Post-processing, validation references and file export.

Derived quantities (vorticity, stream function, forces), the analytical and
published solutions the benchmarks are measured against, and the VTK/CSV/NPZ
writers.

Also the judgement modules, which exist to answer questions a bare number
cannot: whether a mean was worth taking (:mod:`~pycfd.analysis.timeseries`),
whether a wake is really shedding (:mod:`~pycfd.analysis.shedding`), whether a
grid sequence can be extrapolated (:mod:`~pycfd.analysis.richardson`), and
whether the whole setup holds together (:mod:`~pycfd.analysis.diagnose`).
"""
