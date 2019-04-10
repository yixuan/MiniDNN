/// \file
/// Source code file of the EigenStream class, it contains the implementation of
/// several methods for input output operations.

#ifndef EigenStream_H
#define EigenStream_H

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include "M_Assert.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#pragma GCC diagnostic pop
#define MAXBUFSIZE (static_cast<int> (1e6))


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

/// Class for input-output manipulation
class EigenStream
{
    private:

    public:

        //--------------------------------------------------------------------------
        /// Export the matrices in numpy (type=python), matlab (type=matlab) format and txt (type=eigen) format
        /* In this case the function is implemented for a second order matrix  */
        ///
        /// @param[in] matrice Eigen::MatrixXd that you want to export.
        /// @param[in] Name string to identify the name you want to use to save the file.
        /// @param[in] type string to identify format to export the matrix if numpy (type="python"), if matlab (type="matlab") if txt (type="eigen").
        /// @param[in] folder string to identify the folder where you want to save the file.
        ///
        static void exportMatrix(Eigen::MatrixXd& matrix, std::string Name,
                                 std::string type = "python", std::string folder = "./Model")
        {
            mkdir(folder.c_str(), ACCESSPERMS);
            std::string est;

            if (type == "python")
            {
                est = ".py";
                std::string filename(folder + "/" + Name + "_mat" + est);
                std::ofstream str(filename.c_str());
                str << Name << "=np.array([";

                for (int i = 0; i < matrix.rows(); i++)
                {
                    for (int j = 0; j < matrix.cols(); j++)
                    {
                        if (j == 0)
                        {
                            str << "[" << matrix(i, j);
                        }
                        else
                        {
                            str << "," << matrix(i, j);
                        }
                    }

                    if (i != (matrix.rows() - 1))
                    {
                        str << "]," << std::endl;
                    }
                }

                str << "]])" << std::endl;
            }

            if (type == "matlab")
            {
                est = ".m";
                std::string filename(folder + "/" + Name + "_mat" + est);
                std::ofstream str(filename.c_str());
                str << Name << "=[";

                for (int i = 0; i < matrix.rows(); i++)
                {
                    for (int j = 0; j < matrix.cols(); j++)
                    {
                        str << " " << matrix(i, j);
                    }

                    if (i != (matrix.rows() - 1))
                    {
                        str << ";" << std::endl;
                    }
                }

                str << "];" << std::endl;
            }

            if (type == "eigen")
            {
                std::ofstream ofs;
                std::string filename(folder + "/" + Name + "_mat.txt");
                ofs.open(filename.c_str());
                ofs << matrix << std::endl;
                ofs.close();
            }
        }

        //--------------------------------------------------------------------------
        /// @brief      Saves a dense matrix to a binary format file
        ///
        /// @param[in]  Matrix      The  Eigen dense matrix
        /// @param[in]  folder      the folder where you want to save the matrix
        /// @param[in]  MatrixName  The matrix name for the output file
        ///
        /// @tparam     MatrixType  type of the matrix, i.e. double, float, ...
        ///
        template <typename MatrixType>
        static void SaveDenseMatrix(MatrixType& Matrix, std::string folder,
                                    std::string MatrixName)
        {
            mkdir(folder.c_str(), ACCESSPERMS);
            std::ofstream out(folder + MatrixName,
                              std::ios::out | std::ios::binary | std::ios::trunc);
            typename MatrixType::Index rows = Matrix.rows(), cols = Matrix.cols();
            out.write(reinterpret_cast<char*> (&rows), sizeof(typename MatrixType::Index));
            out.write(reinterpret_cast<char*> (&cols), sizeof(typename MatrixType::Index));
            out.write(reinterpret_cast<char*> (Matrix.data()),
                      rows * cols * sizeof(typename MatrixType::Scalar) );
            out.close();
        }

        //--------------------------------------------------------------------------
        /// @brief      Reads a dense matrix from a binary format file
        ///
        /// @param[in,out]  Matrix      The Eigen dense matrix
        /// @param[in]      folder      The folder from where you want to read the matrix
        /// @param[in]      MatrixName  The matrix name of the input file
        ///
        /// @tparam     MatrixType  type of the matrix, i.e. double, float, ...
        ///
        template <typename MatrixType>
        void ReadDenseMatrix(MatrixType& Matrix, std::string folder,
                             std::string MatrixName)
        {
            std::ifstream in;
            in.open((folder + MatrixName).c_str(), std::ios::in | std::ios::binary);
            std::string message(folder + MatrixName +
                                " file does not exist, Check if the file is existing");
            M_Assert(in.good(), message.c_str());

            if (in.is_open())
            {
                typename MatrixType::Index rows = 0, cols = 0;
                in.read(reinterpret_cast<char*> (&rows), sizeof(typename MatrixType::Index));
                in.read(reinterpret_cast<char*> (&cols), sizeof(typename MatrixType::Index));
                Matrix.resize(rows, cols);
                in.read( reinterpret_cast<char*>(Matrix.data()),
                         rows * cols * sizeof(typename MatrixType::Scalar) );
                in.close();
            }
        }

        //----------------------------------------------------------------------
        /// @brief      Saves a dense tensor.
        ///
        /// @param      Tensor      The tensor
        /// @param[in]  folder      The folder
        /// @param[in]  MatrixName  The matrix name
        ///
        /// @tparam     TensorType  type of the tensor, i.e. double, float, ...
        ///
        template <typename TensorType>
        static void SaveDenseTensor(TensorType& Tensor, std::string folder,
                                    std::string MatrixName)
        {
            std::ofstream out(folder + MatrixName,
                              std::ios::out | std::ios::binary | std::ios::trunc);
            typename TensorType::Dimensions dim = Tensor.dimensions();
            int tot = 1;

            for (unsigned int k = 0; k < dim.size(); k++)
            {
                tot *= dim[k];
            }

            out.write(reinterpret_cast<char*> (&dim),
                      sizeof(typename TensorType::Dimensions));
            out.write(reinterpret_cast<char*> (Tensor.data()),
                      tot * sizeof(typename TensorType::Scalar) );
            out.close();
        }

        //----------------------------------------------------------------------
        /// @brief      Reads a dense tensor.
        ///
        /// @param      Tensor      The tensor
        /// @param[in]  folder      The folder
        /// @param[in]  MatrixName  The matrix name
        ///
        /// @tparam     TensorType   type of the tensor, i.e. double, float, ...
        ///
        template <typename TensorType>
        static void ReadDenseTensor(TensorType& Tensor, std::string folder,
                                    std::string MatrixName)
        {
            std::ifstream in;
            in.open((folder + MatrixName).c_str(), std::ios::in | std::ios::binary);
            typename TensorType::Dimensions dim;
            in.read(reinterpret_cast<char*> (&dim),
                    sizeof(typename TensorType::Dimensions));
            auto dims = Tensor.dimensions();
            M_Assert(dims.size() == dim.size(),
                     "The rank of the tensor you want to fill does not coincide with the rank of the tensor you are reading");
            int tot = 1;

            for (unsigned int k = 0; k < dim.size(); k++)
            {
                tot *= dim[k];
            }

            Tensor.resize(dim);
            in.read( reinterpret_cast<char*>(Tensor.data()),
                     tot * sizeof(typename TensorType::Scalar) );
            in.close();
        }




};

#endif






