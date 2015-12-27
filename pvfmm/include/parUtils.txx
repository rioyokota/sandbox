
/**
  @file parUtils.txx
  @brief Definitions of the templated functions in the par module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
  */

#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <dtypes.h>
#include <ompUtils.h>
#include <mem_mgr.hpp>
#include <matrix.hpp>

namespace pvfmm{
namespace par{

  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls,
        T* recvbuf, int* recvcnts, int* rdispls, const MPI_Comm &comm) {

#ifndef ALLTOALLV_FIX
      return MPI_Alltoallv(sendbuf, sendcnts, sdispls, par::Mpi_datatype<T>::value(),
                           recvbuf, recvcnts, rdispls, par::Mpi_datatype<T>::value(), comm);
#else

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      int commCnt = 0;

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = (rank+1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      MPI_Request* requests = mem::aligned_new<MPI_Request>(commCnt);
      assert(requests || !commCnt);

      MPI_Status* statuses = mem::aligned_new<MPI_Status>(commCnt);
      assert(statuses || !commCnt);

      commCnt = 0;

      //First place all recv requests. Do not recv from self.
      for(int i = 0; i < rank; i++) {
        if(recvcnts[i] > 0) {
          MPI_Irecv( &(recvbuf[rdispls[i]]) , recvcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(recvcnts[i] > 0) {
          MPI_Irecv( &(recvbuf[rdispls[i]]) , recvcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Next send the messages. Do not send to self.
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          MPI_Issend( &(sendbuf[sdispls[i]]), sendcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          MPI_Issend( &(sendbuf[sdispls[i]]), sendcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Now copy local portion.
      #pragma omp parallel for
      for(int i = 0; i < sendcnts[rank]; i++) {
        recvbuf[rdispls[rank] + i] = sendbuf[sdispls[rank] + i];
      }

      if(commCnt) MPI_Waitall(commCnt, requests, statuses);

      mem::aligned_delete(requests);
      mem::aligned_delete(statuses);
      return 0;
#endif
    }


  template<typename T>
    int partitionW(Vector<T>& nodeList, long long* wts, const MPI_Comm& comm){

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      long long npesLong = npes;

      long long nlSize = nodeList.Dim();
      long long off1= 0, off2= 0, localWt= 0, totalWt = 0;

      // First construct arrays of wts.
      Vector<long long> wts_(nlSize);
      if(wts == NULL) {
        wts=&wts_[0];
        #pragma omp parallel for
        for (long long i = 0; i < nlSize; i++){
          wts[i] = 1;
        }
      }
      #pragma omp parallel for reduction(+:localWt)
      for (long long i = 0; i < nlSize; i++){
        localWt+=wts[i];
      }

      // compute the total weight of the problem ...
      MPI_Allreduce(&localWt, &totalWt, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
      MPI_Scan(&localWt, &off2, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm );
      off1=off2-localWt;

      // perform a local scan on the weights first ...
      Vector<long long> lscn(nlSize);
      if(nlSize) {
        lscn[0]=off1;
        omp_par::scan(&wts[0],&lscn[0],nlSize);
      }

      Vector<int> int_buff(npesLong*4);
      Vector<int> sendSz (npesLong,&int_buff[0]+npesLong*0,false);
      Vector<int> recvSz (npesLong,&int_buff[0]+npesLong*1,false);
      Vector<int> sendOff(npesLong,&int_buff[0]+npesLong*2,false);
      Vector<int> recvOff(npesLong,&int_buff[0]+npesLong*3,false);

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...

      #pragma omp parallel for
      for (size_t i = 0; i < npesLong; i++) {
        sendSz[i] = 0;
      }

      //The Heart of the algorithm....
      if(nlSize>0 && totalWt>0) {
        long long pid1=( off1   *npesLong)/totalWt;
        long long pid2=((off2+1)*npesLong)/totalWt+1;
        assert((totalWt*pid2)/npesLong>=off2);
        pid1=(pid1<       0?       0:pid1);
        pid2=(pid2>npesLong?npesLong:pid2);
        #pragma omp parallel for
        for(int i=pid1;i<pid2;i++){
          long long wt1=(totalWt*(i  ))/npesLong;
          long long wt2=(totalWt*(i+1))/npesLong;
          long long start = std::lower_bound(&lscn[0], &lscn[0]+nlSize, wt1, std::less<long long>())-&lscn[0];
          long long end   = std::lower_bound(&lscn[0], &lscn[0]+nlSize, wt2, std::less<long long>())-&lscn[0];
          if(i==         0) start=0     ;
          if(i==npesLong-1) end  =nlSize;
          sendSz[i]=end-start;
        }
      }else sendSz[0]=nlSize;

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      MPI_Alltoall(&sendSz[0], 1, par::Mpi_datatype<int>::value(),
          &recvSz[0], 1, par::Mpi_datatype<int>::value(), comm);

      // compute offsets ...
      sendOff[0] = 0; omp_par::scan(&sendSz[0],&sendOff[0],npesLong);
      recvOff[0] = 0; omp_par::scan(&recvSz[0],&recvOff[0],npesLong);

      // new value of nlSize, ie the local nodes.
      long long nn = recvSz[npesLong-1] + recvOff[npes-1];

      return 0;
    }//end function

  template<typename T>
    int partitionW(std::vector<T>& nodeList, long long* wts, const MPI_Comm& comm){
      Vector<T> nodeList_=nodeList;
      int ret = par::partitionW<T>(nodeList_, wts, comm);

      nodeList.assign(&nodeList_[0],&nodeList_[0]+nodeList_.Dim());
      return ret;
    }


  template<typename T>
    int HyperQuickSort(const Vector<T>& arr_, Vector<T>& SortedElem){

      // Get comm size and rank.
      int npes=1, myrank=0;
      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      long long totSize, nelem = arr_.Dim();
      totSize = nelem;

      // Local sort.
      Vector<T> arr=arr_;
      omp_par::merge_sort(&arr[0], &arr[0]+nelem);

      // Allocate memory.
      //Vector<T> nbuff;
      //Vector<T> nbuff_ext;
      //Vector<T> rbuff    ;
      //Vector<T> rbuff_ext;

      SortedElem.Resize(nelem);
      memcpy(&SortedElem[0], &arr[0], nelem*sizeof(T));

      return 0;
    }//end function

  template<typename T>
    int HyperQuickSort(const std::vector<T>& arr_, std::vector<T>& SortedElem_){
      Vector<T> SortedElem;
      const Vector<T> arr(arr_.size(),(T*)&arr_[0],false);

      int ret = HyperQuickSort(arr, SortedElem);
      SortedElem_.assign(&SortedElem[0],&SortedElem[0]+SortedElem.Dim());
      return ret;
    }


  template<typename T>
    int SortScatterIndex(const Vector<T>& key, Vector<size_t>& scatter_index, const T* split_key_){
      typedef SortPair<T,size_t> Pair_t;

      int npes=1, rank=0;
      long long npesLong = npes;

      Vector<Pair_t> parray(key.Dim());
      { // Build global index.
        long long glb_dsp=0;
        long long loc_size=key.Dim();
        glb_dsp=0;
        #pragma omp parallel for
        for(size_t i=0;i<loc_size;i++){
          parray[i].key=key[i];
          parray[i].data=glb_dsp+i;
        }
      }

      Vector<Pair_t> psorted;
      HyperQuickSort(parray, psorted);

      scatter_index.Resize(psorted.Dim());

      #pragma omp parallel for
      for(size_t i=0;i<psorted.Dim();i++){
        scatter_index[i]=psorted[i].data;
      }

      return 0;
    }

  template<typename T>
    int ScatterForward(Vector<T>& data_, const Vector<size_t>& scatter_index){
      typedef SortPair<size_t,size_t> Pair_t;

      int npes=1, rank=0;
      long long npesLong = npes;

      size_t data_dim=0;
      long long send_size=0;
      long long recv_size=0;
      {
        recv_size=scatter_index.Dim();
        long long loc_size[2]={(long long)(data_.Dim()*sizeof(T)), recv_size};
        if(loc_size[0]==0 || loc_size[1]==0) return 0; //Nothing to be done.
        data_dim=loc_size[0]/loc_size[1];
        send_size=(data_.Dim()*sizeof(T))/data_dim;
      }

      Vector<char> recv_buff(recv_size*data_dim);

      // Sort scatter_index.
      Vector<Pair_t> psorted(recv_size);
      {
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          psorted[i].key=scatter_index[i];
          psorted[i].data=i;
        }
        omp_par::merge_sort(&psorted[0], &psorted[0]+recv_size);
      }

       // Prepare send buffer
      {
        char* data=(char*)&data_[0];
        #pragma omp parallel for
        for(size_t i=0;i<send_size;i++){
          size_t src_indx=psorted[i].key*data_dim;
          size_t trg_indx=i*data_dim;
          for(size_t j=0;j<data_dim;j++) {
            recv_buff[trg_indx+j]=data[src_indx+j];
	  }
        }
      }

      // Build output data.
      {
        char* data=(char*)&data_[0];
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          size_t src_indx=i*data_dim;
          size_t trg_indx=psorted[i].data*data_dim;
          for(size_t j=0;j<data_dim;j++)
            data[trg_indx+j]=recv_buff[src_indx+j];
        }
      }
      return 0;
    }

}//end namespace
}//end namespace
