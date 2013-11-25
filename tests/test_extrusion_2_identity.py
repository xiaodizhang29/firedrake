"""Tests for successful identity solve on extruded meshes"""
import numpy as np
import pytest

from firedrake import *
from common import *

CG = [("CG", 1), ("CG", 2)]
DG = [("DG", 0), ("DG", 1)]
hdiv = [("RT", 1), ("RT", 2), ("RT", 3), ("BDM", 1)]
params = {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}


@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG + DG for (f, d) in CG + DG])
def test_identity_scalar(hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)
    fspace = FunctionSpace(mesh, hfamily, hdegree, vfamily=vfamily, vdegree=vdegree)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    f = project(Expression("x[2]-x[0]"), fspace)

    out = Function(fspace)
    solve(u*v*dx == f*v*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-14


@pytest.mark.xfail(reason="No support for vector function spaces on extruded meshes")
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG + DG for (f, d) in CG + DG])
def test_identity_vector(hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)
    fspace = VectorFunctionSpace(mesh, hfamily, hdegree, vfamily=vfamily, vdegree=vdegree)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    f = project(Expression(("x[2]-x[0]", "x[1] - x[2]", "x[0] - x[1]")), fspace)

    out = Function(fspace)
    solve(u*v*dx == f*v*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-14


# two valid combinations for hdiv: 1) BDM/RT x DG, 2) DG x CG
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in DG])
def test_identity_hdiv(hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HDiv(OuterProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    f = project(Expression(("x[1]", "-x[0]", "x[2]")), fspace)

    out = Function(fspace)
    solve(dot(u, v)*dx == dot(f, v)*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-14


# two valid combinations for hcurl: 1) BDM/RT x CG, 2) CG x DG
@pytest.mark.parametrize(('hfamily', 'hdegree', 'vfamily', 'vdegree'),
                         [(f, d, vf, vd) for (vf, vd) in CG for (f, d) in hdiv]
                         + [(f, d, vf, vd) for (vf, vd) in DG for (f, d) in CG])
def test_identity_hcurl(hfamily, hdegree, vfamily, vdegree):
    mesh = extmesh(4, 4, 2)

    horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HCurl(OuterProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    f = project(Expression(("x[1]", "-x[0]", "x[2]")), fspace)

    out = Function(fspace)
    solve(dot(u, v)*dx == dot(f, v)*dx, out, solver_parameters=params)
    assert np.max(np.abs(out.dat.data - f.dat.data)) < 1.0e-14

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))