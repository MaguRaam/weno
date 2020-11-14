#include "../include/stencil.h"

namespace weno
{
    template <int dim>
    void Stencil<dim>::initialize(const dealii::DoFHandler<dim> &dof_handler)
    {
        /*
         * Mark locally relevant cells.
           locally relevant cells for our purpose = locally owned cells + 1 layers of face sharing ghost cells:
         * 
         *  Index locally relevant cells using a local index.
         * 
         *  establish map between local cell index <-> global cell index <-> cell iterator for the relevant cells.
         * 
         *  also map local to global vertex and face index for relevant cells:
         */

        //initialize global index function object:
        GlobalIndex<dim> global_index;

        //create a big vector of bools equal to total no of cells that marks a cell as relevant or not:
        std::vector<bool> is_relevant_cell(dof_handler.n_dofs(), false); //initialize with false:

        //set of global vertex and face indices:
        std::set<unsigned int> vertex_index;
        std::set<unsigned int> face_index;

        //initialize and increment the local index:
        unsigned int local_index = 0;

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            //locally owned cells:
            if (cell->is_locally_owned())
            {
                //mark locally owned cell is relevant:
                is_relevant_cell[global_index(cell)] = true;

                //between local cell index <-> global cell index <-> cell iterator
                global_to_local_index_map[global_index(cell)] = local_index;
                local_to_global_index_map.push_back(global_index(cell));
                local_index_to_iterator.push_back(cell);

                local_index++;

                //compute the global vertex index of the cell and insert in vertex index set:
                for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                    vertex_index.insert(cell->vertex_index(i));

                //similarly face index:
                for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
                {
                    face_index.insert(cell->face_index(f));

                    //mark one layer of ghost cell as relevant:
                    if (!cell->face(f)->at_boundary())
                    {
                        auto neighbor = cell->neighbor(f);
                        is_relevant_cell[global_index(neighbor)] = true;
                    }
                }
            }
        }

        /*
         * mark extra layer of ghost cells (3 or more layers)
         * 
         * also establish connectivity between local cell index <-> global cell index <-> cell iterator
           for that 1  layer of ghost cell.
         */

        //extra layer of ghost cells indicator:
        std::vector<bool> is_extra_ghost_cell(dof_handler.n_dofs(), false);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            //check for only the 1st layer of ghost cells:
            if (is_relevant_cell[global_index(cell)] && !(cell->is_locally_owned()))
            {
                //establish connectivity between local cell index <-> global cell index <-> cell iterator
                is_relevant_cell[global_index(cell)] = true;
                global_to_local_index_map[global_index(cell)] = local_index;
                local_to_global_index_map.push_back(global_index(cell));
                local_index_to_iterator.push_back(cell);
                local_index++;

                //loop over faces of the cells in ghost layer
                for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
                {
                    //interior faces:
                    if (!cell->face(f)->at_boundary())
                    {
                        //neighbor cell iterator:
                        auto neighbor = cell->neighbor(f);

                        if (!neighbor->is_locally_owned())
                        {
                            is_extra_ghost_cell[global_index(neighbor)] = true;
                            for (unsigned int f1 = 0; f1 < dealii::GeometryInfo<dim>::faces_per_cell; ++f1)
                            {
                                if (!neighbor->face(f1)->at_boundary())
                                {
                                    auto neighbor1 = neighbor->neighbor(f1);
                                    if (!neighbor1->is_locally_owned())
                                    {
                                        is_extra_ghost_cell[global_index(neighbor1)] = true;
                                        for (unsigned int f2 = 0; f2 < dealii::GeometryInfo<dim>::faces_per_cell; ++f2)
                                        {
                                            if (!neighbor1->face(f2)->at_boundary())
                                            {
                                                auto neighbor2 = neighbor1->neighbor(f2);
                                                if (!neighbor2->is_locally_owned())
                                                {
                                                    is_extra_ghost_cell[global_index(neighbor2)] = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        //n_relevant_cells = locally owned cells + 1 layer of face sharing ghost cells:
        n_relevant_cells = local_index;

        //collect the vetices of extra ghost cells and insert in vertex index set:
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            if (is_extra_ghost_cell[global_index(cell)])
                for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                    vertex_index.insert(cell->vertex_index(i));
        }

        //total no of vertices including extra ghost cell vertices:
        unsigned int n_vertices = vertex_index.size();
        //unsigned int n_faces = face_index.size();

        //map between global to local vertex index:
        std::map<unsigned int, unsigned int> vertex_map;

        unsigned int local_vertex = 0;
        for (const auto &v : vertex_index)
        {
            vertex_map[v] = local_vertex;
            local_vertex++;
        }

        //map between global to local face index:
        std::map<unsigned int, unsigned int> face_index_map;

        unsigned int local_face_index = 0;
        for (const auto &f : face_index)
        {
            face_index_map[f] = local_face_index;
            local_face_index++;
        }

        /*
         * Establish vertex to cell connectivity for locally owned cells + extra 3 layers of ghost cells
         * given local vertex index return cells(cell-iterators) sharing the vertex.
         */
        std::vector<std::set<active_cell_iterator>> vertex_to_cell_iterator(n_vertices);

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            //locally owned cells + extra_ghost_cells:
            if (cell->is_locally_owned() || is_extra_ghost_cell[global_index(cell)])
            {
                //loop over vertices in the current cell:
                for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_cell; ++i)
                {
                    //add the current cell in the appropriate vertex indices in the vertex_to_cell_iterator vector:
                    vertex_to_cell_iterator[vertex_map[cell->vertex_index(i)]].insert(cell);
                }
            }
        }

        //initialize and find the stencils:
        cell_neighbor_iterator.resize(n_relevant_cells);
        cell_all_neighbor_iterator.resize(n_relevant_cells); //TODO Ask Dipak do we need this?
        cell_diagonal_neighbor_iterator.resize(n_relevant_cells);
        cell_neighbor_neighbor_iterator.resize(n_relevant_cells);
        cell_face_neighbor_iterator.resize(n_relevant_cells);

        for (unsigned int i = 0; i < n_relevant_cells; i++)
        {
            cell_neighbor_iterator[i].resize(dealii::GeometryInfo<dim>::faces_per_cell);
            cell_neighbor_neighbor_iterator[i].resize(dealii::GeometryInfo<dim>::faces_per_cell);
        }

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            //check for relevant cells:
            if (is_relevant_cell[global_index(cell)])
            {

                //local index of the cell:
                unsigned int local_i = global_to_local_index_map[global_index(cell)];

                //loop over faces:
                for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
                {

                    //loop over vertices in the face:
                    for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::vertices_per_face; ++i)
                    {
                        //get first and last cell itertors of the vertex sharing cells:
                        auto adjacent_cell = vertex_to_cell_iterator[vertex_map[cell->face(f)->vertex_index(i)]].begin();
                        auto end_cell = vertex_to_cell_iterator[vertex_map[cell->face(f)->vertex_index(i)]].end();

                        //loop over vertex sharing cells:
                        for (; adjacent_cell != end_cell; ++adjacent_cell)
                        {

                            //dump these cells in our cell neighbor containers:
                            cell_neighbor_iterator[local_i][f].insert(*adjacent_cell);
                            cell_all_neighbor_iterator[local_i].insert(*adjacent_cell);
                            cell_diagonal_neighbor_iterator[local_i].insert(*adjacent_cell);
                        }
                    }

                    //insert neighbor neighbor cell in the cell_neighbor_neighbor_iterator

                    //interior face
                    if (!cell->face(f)->at_boundary())
                    {
                        //neighbor cell:
                        auto neighbor = cell->neighbor(f);

                        //insert cell neighbors in the cell_face_neighbor_iterator:
                        cell_face_neighbor_iterator[local_i].insert(neighbor);

                        //the current face index relative to the neighbor cell:
                        unsigned int neighbor_face_index = cell->neighbor_of_neighbor(f);

                        unsigned int neighbor_neighbor_face_index = dealii::numbers::invalid_unsigned_int;

                        //identifying neighbor neighbor face index:
                        if (neighbor_face_index % 2 == 0)
                            neighbor_neighbor_face_index = neighbor_face_index + 1;
                        else
                            neighbor_neighbor_face_index = neighbor_face_index - 1;

                        //insert the neighbor neighbor cell:
                        if (!neighbor->face(neighbor_neighbor_face_index)->at_boundary())
                            cell_neighbor_neighbor_iterator[local_i][f].insert(neighbor->neighbor(neighbor_neighbor_face_index));
                    }

                } //end of face loop

                //loop over faces:
                for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
                {
                    //index opposite face of f is ff:
                    unsigned int ff = dealii::numbers::invalid_unsigned_int;

                    if (f % 2 == 0)
                        ff = f + 1;
                    else
                        ff = f - 1;

                    //loop over faces except the ff face and remove face cells to get vornoi neighbor:
                    for (unsigned int f3 = 0; f3 < dealii::GeometryInfo<dim>::faces_per_cell; ++f3)
                    {
                        if (f3 != ff)
                            if (!cell->face(f3)->at_boundary())
                                cell_neighbor_iterator[local_i][f].erase(cell->neighbor(f3));
                    }

                    //again remove face neigbhor cells to retain vornoi neighbors:
                    if (!cell->face(f)->at_boundary())
                        cell_diagonal_neighbor_iterator[local_i].erase(cell->neighbor(f));

                    //remove centre cell in the stencil:
                    cell_neighbor_iterator[local_i][f].erase(cell);
                }

                //remove centre cell in the stencil:
                cell_all_neighbor_iterator[local_i].erase(cell);
                cell_diagonal_neighbor_iterator[local_i].erase(cell);
            }
        }
    }

    //explicit instantiation
    template class Stencil<1>;
    template class Stencil<2>;
    template class Stencil<3>;

} // namespace weno
