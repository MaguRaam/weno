#include "stencil.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// make grid:
template <int dim>
inline void make_grid(Triangulation<dim> &tria)
{
  GridGenerator::subdivided_hyper_cube(tria, 4, -1.0, 1.0, true);
}

int main(int argc, char *argv[])
{
  // problem dimension:
  const int dim = 3;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  //mpi communicator object:
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  //parallel cout:
  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  //print nprocess:
  pcout << "==============================================" << std::endl;
  pcout << "Running on " << Utilities::MPI::n_mpi_processes(mpi_communicator) << " MPI rank(s)..." << std::endl;

  // shared triangulation:
  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD,
                                            typename Triangulation<dim>::MeshSmoothing(Triangulation<dim>::none), false,
                                            parallel::shared::Triangulation<dim>::Settings::partition_zorder);
  //make grid:
  make_grid(tria);

  //memory consumed by the grid:
  std::size_t memory = tria.memory_consumption();
  pcout << "=============================================" << std::endl;
  pcout << "Total number of cells: " << tria.n_active_cells() << std::endl;
  pcout << "memory in GB: " << memory / std::pow(2, 30) << std::endl;

  //enumerate cells:
  DoFHandler<dim> dof_handler(tria);
  FE_DGQ<dim> fv(0);
  dof_handler.distribute_dofs(fv);

  //stencil:
  weno::Stencil<dim> stencil;
  stencil.initialize(dof_handler);

  for (unsigned int i = 0; i < stencil.no_relevant_cells(); i++)
  {
    auto central_stencil = stencil.central_stencil_constrained(i);
    std::cout<<central_stencil.size()<<std::endl;
  }

  return 0;
}
