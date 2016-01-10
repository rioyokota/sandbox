/**
  @file parUtils.h
  @brief A set of parallel utilities.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
  @author Dhairya Malhotra, dhairya.malhotra@gmail.com
  */

//#include <mpi.h>
#include <vector>
#include <cstdlib>

#include <vector.hpp>

#ifndef __PVFMM_PAR_UTILS_H_
#define __PVFMM_PAR_UTILS_H_

/**
  @namespace par
  @author Rahul Sampath
  @author Hari Sundar
  @brief Collection of Generic Parallel Functions: Sorting, Partitioning, Searching,...
  */
namespace pvfmm{
namespace par{

  template<typename T>
    int HyperQuickSort(const Vector<T>& in, Vector<T> & out);
  template<typename T>
    int HyperQuickSort(const std::vector<T>& in, std::vector<T> & out);

  template<typename A, typename B>
    struct SortPair{
      int operator<(const SortPair<A,B>& p1) const{ return key<p1.key;}
      A key;
      B data;
    };

  template<typename T>
    int SortScatterIndex(const Vector<T>& key, Vector<size_t>& scatter_index,
        const T* split_key=NULL);
  template<typename T>
    int ScatterForward(Vector<T>& data, const Vector<size_t>& scatter_index);

}//end namespace
}//end namespace

#include "parUtils.txx"

#endif //__PVFMM_PAR_UTILS_H_
