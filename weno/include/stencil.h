/**
 * @file stencil.h
 * @author Dipak
 * @brief 
 * @version 0.1
 * @date 2020-11-05
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef weno_stencil_h
#define weno_stencil_h

#include <deal.II/dofs/dof_handler.h>

namespace weno
{
    //Forward declaration
    template <int dim>
    struct GlobalIndex;

    /**
     * @brief Given a dealii's parallel::shared::Triangulation<dim> 
     * TODO
     * 
     * @tparam dim 
     */
    template <int dim>
    class Stencil
    {
    public:
        /**
         * @brief spatial dimension
         * 
         */
        static constexpr unsigned int dimension = dim;

        /**
         * @brief active cell iterator type
         * 
         */
        using active_cell_iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

        /**
         * @brief default constructor does not initialize any data. After constructing 
         * an object with this constructor, use initialize() to make a valid Stencil.
         * 
         */
        Stencil() = default;

        /**
         * @brief takes DoFHandler object to initialize private data members.
         * 
         * @param dof_handler 
         */
        void initialize(const dealii::DoFHandler<dim> &dof_handler);

        /**
         * @brief deleted copy constructor for this class to avoid expensive copy
         * 
         */
        Stencil(const Stencil &) = delete;

        /**
         * @brief move constructor to steal data from temporary object
         * 
         */
        Stencil(Stencil &&) = default;

        /**
         * @brief deleted copy assignment operator to avoid expensive copy
         *  
         */
        Stencil &operator=(const Stencil &) = delete;

        /**
         * @brief move assignment to steal data from temporary object
         */
        Stencil &operator=(Stencil &&) = default;

        /**
         * @brief Destroy the stencil object
         * 
         */
        ~Stencil() = default;

        /**
         * @brief the function returns the no of relevant cells:
         * n_relevant_cells = locally owned cells + 1 layer of face sharing ghost cells
         * @return unsigned int
         */
        unsigned int no_relevant_cells() const
        {
            return n_relevant_cells;
        }

        /**
         * @brief Given the locally relevant cell index, the function returns the constrained 
         * part of the central stencil. These are simply the face neighbors of the cell.
         * 
         * For example in 2d the constrained part of central stencil looks like.
         * 
         * 
         *          *---*
         *          | c |
         *      *---*---*---*
         *      | c |   | c |
         *      *---*---*---*
         *          | c |
         *          *---*
         *
         * 
         * 
         * @param relevant_cell_index 
         * @return std::set<active_cell_iterator> 
         */
        const std::set<active_cell_iterator> &central_stencil_constrained(unsigned int relevant_cell_index) const
        {
            return cell_face_neighbor_iterator[relevant_cell_index];
        }

    private:
        /**
         * @brief no of relevant cells = locally owned cells + 1 layer of face sharing ghost cells:
         */
        unsigned int n_relevant_cells;

        /**
         * @brief global to local index for the relevant cells
         */
        std::map<unsigned int, unsigned int> global_to_local_index_map;

        /**
         * @brief local to global index for the relevant cells
         */
        std::vector<unsigned int> local_to_global_index_map;

        /**
         * @brief local index to cell iterator
         * 
         */
        std::vector<active_cell_iterator> local_index_to_iterator;

        /**
         * @brief cell_face neighbor_iterator:
         */
        std::vector<std::set<active_cell_iterator>> cell_face_neighbor_iterator;

        /**
         * @brief voronoi neighbour for directional stencils 
         */
        std::vector<std::vector<std::set<active_cell_iterator>>> cell_neighbor_iterator;

        /**
         * @brief second layer face neighbour
         */
        std::vector<std::vector<std::set<active_cell_iterator>>> cell_neighbor_neighbor_iterator;

        /**
         * @brief everything except central cell and neighbor-neighbor cell
         */
        std::vector<std::set<active_cell_iterator>> cell_all_neighbor_iterator;

        /**
         * @brief vornoi neighbor for cental stencil
         */
        std::vector<std::set<active_cell_iterator>> cell_diagonal_neighbor_iterator;
    };

    /**---------------------------------------------------------------------**/

    /**
     * @brief Global Index functor 
     * 
     * @tparam dim 
     */
    template <int dim>
    struct GlobalIndex
    {
        /**
         * @brief Construct a new Global Index object
         * 
         */
        GlobalIndex() : local_dof_indices(1) {}

        /**
         * @brief function call operator takes dealii::DoFHandler<dim>::active_cell_iterator 
         * as input and returns global index of the cell
         * 
         * @param cell 
         * @return dealii::types::global_dof_index 
         */
        dealii::types::global_dof_index operator()(const typename dealii::DoFHandler<dim>::active_cell_iterator &cell)
        {
            cell->get_dof_indices(local_dof_indices);
            return local_dof_indices[0];
        }

    private:
        std::vector<dealii::types::global_dof_index> local_dof_indices;
    };

} // namespace weno

#endif
