==============================
 Demo of GenEO preconditioner
==============================

::

    from firedrake import *
    from firedrake import dmplex
    from firedrake.petsc import PETSc
    from slepc4py import SLEPc
    from mpi4py import MPI
    import numpy
    from pyop2.datatypes import IntType
    mesh = UnitSquareMesh(2,2, distribution_parameters={"overlap_type":
                                                        (DistributedMeshOverlapType.FACET, 1)})

    V = FunctionSpace(mesh, "P", 1)

    u = TrialFunction(V)

    v = TestFunction(V)

    a = u*v*dx

    A = assemble(a, mat_type="is")

    A.M.handle.view()
    print(A.M.handle.getSizes())
    #import sys; sys.exit()

    which = numpy.unique(V.cell_node_map().values)

    sf = V.dm.getDefaultSF()

    _, nleaves, leaves, _ = sf.getGraph()
    selected = numpy.arange(nleaves, dtype=IntType)[numpy.in1d(leaves, which)]
    sf = sf.createEmbeddedLeafSF(selected)
    sf.view()
    degree = sf.computeDegree()
    comm = mesh.comm
    maxdegree = numpy.asarray([degree.max()])

    lgmap = V.dof_dset.lgmap
    comm.Allreduce(MPI.IN_PLACE, maxdegree, op=MPI.MAX)

    maxdegree, = maxdegree
    sf.view()
    PETSc.Sys.syncPrint("[%d] %s %s" % (mesh.comm.rank, degree, maxdegree))

    PETSc.Sys.syncFlush()

    leafdata = numpy.full(V.node_set.total_size, comm.rank, dtype=IntType)
    rootdata = numpy.full(sum(degree), -1, dtype=IntType)

    dmplex.gatherBegin(sf, MPI.INT, leafdata, rootdata)
    dmplex.gatherEnd(sf, MPI.INT, leafdata, rootdata)

    data = numpy.full((V.node_set.total_size, maxdegree), -1, dtype=IntType)

    offset = 0
    i = 0

    for i, d in enumerate(degree):
        data[i, :d]  = rootdata[offset:offset+d]
        offset += d

    ghosted_roots = op2.Dat(V.node_set**maxdegree, data=data)
    ghosted_roots.halo_valid = False
    PETSc.Sys.syncPrint("[%d] %s" % (mesh.comm.rank, rootdata))
    PETSc.Sys.syncPrint("[%d] %s %s" % (mesh.comm.rank, which, ghosted_roots.data_ro_with_halos))

    intersections = [[] for _ in range(comm.size)]

    for s in selected:
        d = ghosted_roots.data_ro_with_halos[s, ...]
        for dof in d:
            if dof == -1 or dof == comm.rank:
                continue
            intersections[dof].append(s)

    for i in range(len(intersections)):
        intersections[i] = PETSc.IS().createGeneral(lgmap.apply(intersections[i]), comm=COMM_SELF)

    PETSc.Sys.syncPrint("[%d] %s" % (comm.rank, intersections))
    PETSc.Sys.syncFlush()

    leaf_multiplicities = numpy.empty(V.dof_dset.total_size, dtype=IntType)

    dmplex.bcastBegin(sf, MPI.INT, degree, leaf_multiplicities)
    dmplex.bcastEnd(sf, MPI.INT, degree, leaf_multiplicities)
    leaf_multiplicities = leaf_multiplicities[which]



    class GeneoPC(PCBase):

        def initialize(self, pc):
            A, P = pc.getOperators()
            ctx = P.getPythonContext()
            mesh = ctx.a.ufl_domain()
            dm = mesh._plex
            if V.value_size > 1:
                raise NotImplementedError

            P = assemble(ctx.a, bcs=ctx.row_bcs, mat_type="is").M.handle
            # Use SFComputeDegree + GatherBegin/End I think.
            ipc = PETSc.PC().create(comm=pc.comm)
            ipc.setOptionsPrefix("geneo_")
            ipc.setOperators(P, P)
            ipc.setType("geneo")
            multiplicities = PETSc.IS().createGeneral(leaf_multiplicities, comm=COMM_SELF)
            dmplex.setupgeneopc(ipc, multiplicities, intersections)
            ipc.setFromOptions()
            ipc.incrementTabLevel(1, parent=pc)
            self.ipc = ipc

        def update(self, pc):
            pass

        def apply(self, pc, x, y):
            self.ipc.apply(x, y)

        def applyTranspose(self, pc, x, y):
            self.ipc.applyTranspose(x, y)

        def view(self, pc, viewer=None):
            super().view(viewer)
            viewer.printfASCII("GENEO preconditioner:\n")
            self.ipc.view(viewer)


    uh = Function(V)
    solve(a == v*dx, uh, options_prefix="", solver_parameters={"mat_type": "matfree",
                                                               "pc_type": "python",
                                                               "pc_python_type": "__main__.GeneoPC",
                                                               "ksp_initial_guess_nonzero": True})

   
