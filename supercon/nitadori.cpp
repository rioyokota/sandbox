#include <sc_header.hpp>
#include <cstring>
#include <cassert>
#include <cfloat>
#include <algorithm>
#include <vector>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#ifdef __CLANG_FUJITSU
#include <fj_tool/fapp.h>
#else
#define fapp_start(name, number, level)
#define fapp_stop(name, number, level)
#endif

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct Point{
  using Float = float;
  Float x, y;
  int id;

  Point(){}

  Point(double xd, double yd, int id)
    : x(xd), y(yd), id(id)
  {
    if(x >= Float(1.0)) x = std::nextafter(Float(1.0), Float(0.0));
    if(y >= Float(1.0)) y = std::nextafter(Float(1.0), Float(0.0));
  }

  static Float dist2(const Point &p1, const Point &p2){
    Float dx = p1.x - p2.x;
    Float dy = p1.y - p2.y;
    return dx*dx + dy*dy;
  }

  static double dist2(const Point &p1, const Point &p2, const double *X, const double *Y){
    double dx = X[p1.id] - X[p2.id];
    double dy = Y[p1.id] - Y[p2.id];
    return dx*dx + dy*dy;
  }
};

struct Offset2D{
  int major;
  int minor;

  int nbins;
  int *off;

  Offset2D(int major, int minor)
    : major(major), minor(minor)
  {
    nbins = major * minor;

    off = (int *)malloc((nbins*nbins +1) * sizeof(int));
  }

  ~Offset2D(){
    free(off);
  }

  int &loc(int ix, int iy){
    return off[iy + nbins * ix];
  }

  int &loc3(int xh, int xlyh, int yl){
    return off[yl + major * (xlyh + (minor*minor) * xh)];
  }

  int &loc4(int xh, int xl, int yh, int yl){
    return off[yl + major * (yh + minor * (xl + minor * xh))];
  }

  int bin_index(float f){
    int index = (int)(nbins * f);
    assert(0 <= index && index < nbins);
    return index;
  }

  int bin_index(double d){
    float f = d;
    if(f >= 1.0f){
      f = nextafterf(1.0f, 0.0f);
    }
    return bin_index(f);
  }

  int index_xh(double x){
    return bin_index(x) / minor;
  }

  int index_xh(float x){
    return bin_index(x) / minor;
  }

  int index_xlyh(float x, float y){
    int xl = bin_index(x) % minor;
    int yh = bin_index(y) / major;
    return yh + minor *  xl;
  }

  int index_yl(float y){
    return bin_index(y) % major;
  }
};

struct Pair2{
  int i, j;
  double dist2;

  Pair2(){}

  Pair2(const Point &p1, const Point &p2, const double *X, const double *Y){
    i = std::min(p1.id, p2.id);
    j = std::max(p1.id, p2.id);
    dist2 = Point::dist2(p1, p2, X, Y);
  }

  Pair2(const int ii, const int jj, const double r2){
    i = std::min(ii, jj);
    j = std::max(ii, jj);
    dist2 = r2;
  }

  bool operator<(const Pair2 &rhs) const {
    return dist2 < rhs.dist2;
  }

  size_t hash(size_t seed) const {
    int itmp;
    float ftmp = dist2;
    memcpy(&itmp, &ftmp, 4);

    hash_combine(seed, i);
    hash_combine(seed, j);
    hash_combine(seed, itmp);

    return seed;
  }
};

inline void search_nnb(
		       const double h2,
		       const float h2f,
		       const int ibeg, const int iend,
		       const int cend,
		       const int rbeg, const int rend,
		       const Point  * __restrict P,
		       const double * __restrict X,
		       const double * __restrict Y,
		       std::vector<Pair2> &nblist)
{
  for(int i=ibeg; i<iend; i++){
    for(int j=i+1; j<cend; j++){
      float dist2 = P->dist2(P[i], P[j]);
      if(dist2 < h2f && P->dist2(P[i], P[j], X, Y) < h2){
	nblist.emplace_back(P[i], P[j], X, Y);
      }
    }
    for(int j=rbeg; j<rend; j++){
      float dist2 = P->dist2(P[i], P[j]);
      if(dist2 < h2f && P->dist2(P[i], P[j], X, Y) < h2){
	nblist.emplace_back(P[i], P[j], X, Y);
      }
    }
  }
}

inline void search_nnb2(
			const double h2,
			const float h2f,
			const int ibeg, const int iend,
			const int cend,
			const int rbeg, const int rend,
			const Point  * __restrict P,
			const double * __restrict X,
			const double * __restrict Y,
			std::vector<Pair2> &nblist)
{
  static const int kmax = 64;
  float xj[kmax];
  float yj[kmax];
  int   jj[kmax];
  int flags[kmax];

  for(int i=ibeg; i<iend; i++){
    int nk=0;
    const float xi = P[i].x;
    const float yi = P[i].y;
    const int   ii = P[i].id;

    for(int j=i+1; j<cend; j++){
      xj[nk] = P[j].x;
      yj[nk] = P[j].y;
      jj[nk] = P[j].id;
      nk++;
    }
    for(int j=rbeg; j<rend; j++){
      xj[nk] = P[j].x;
      yj[nk] = P[j].y;
      jj[nk] = P[j].id;
      nk++;
    }
    assert(nk < kmax);

    for(int k=0; k<nk; k++){
      float dx = xj[k] - xi;
      float dy = yj[k] - yi;
      float r2 = dx*dx + dy*dy;
      flags[k] = r2 < h2f;
    }
    for(int k=0; k<nk; k++){
      if(flags[k]){
	double dX = X[jj[k]] - X[ii];
	double dY = Y[jj[k]] - Y[ii];
	double dist2 = dX*dX + dY*dY;
	if(dist2 < h2){
	  nblist.emplace_back(ii, jj[k], dist2);
	}
      }
    }
  }
}
#ifdef __ARM_FEATURE_SVE
inline void search_nnb3(
			const double h2,
			const float h2f,
			const int ibeg, const int iend,
			const int cend,
			const int rbeg, const int rend,
			const Point  * __restrict P,
			const double * __restrict X,
			const double * __restrict Y,
			std::vector<Pair2> &nblist)
{
  svfloat32_t h2v = svdup_f32(h2f);
  auto search = [&](int i, int js, int je, int &nj, int jbuf[]){
    svfloat32_t xi = svdup_f32(P[i].x);
    svfloat32_t yi = svdup_f32(P[i].y);
    for(int j=js; j<je; j+=svcntw()){
      svbool_t pg = svwhilelt_b32(j, je);
      svfloat32x3_t pp = svld3_f32(pg, (float *)&P[j]);
      svfloat32_t xj = svget3(pp, 0);
      svfloat32_t yj = svget3(pp, 1);
      svint32_t jj = svindex_s32(j, 1);

      svfloat32_t dx = svsub_x(pg, xj, xi);
      svfloat32_t dy = svsub_x(pg, yj, yi);
      svfloat32_t r2 = svmul_x(pg, dx, dx);
      r2 = svmla_x(pg, r2, dy, dy);

      svbool_t cmp = svcmplt(pg, r2, h2v);
      int      cnt = svcntp_b32(pg, cmp);
      svint32_t jcomp = svcompact(cmp, jj);
      svbool_t    mask  = svwhilelt_b32(0, cnt);  

      svst1_s32(mask, jbuf+nj, jcomp);
      nj += cnt;
    }
  };
  auto search2 = [&](int i, int js1, int je1, int js2, int je2, int &nj, int jbuf[]){
    svfloat32_t xi = svdup_f32(P[i].x);
    svfloat32_t yi = svdup_f32(P[i].y);
    int cnt1 = je1 - js1;
    int cnt2 = je2 - js2;

    svbool_t pg1 = svwhilelt_b32(0, cnt1);
    svbool_t pg2 = svwhilelt_b32(0, cnt2);

    svfloat32x3_t pp1 = svld3_f32(pg1, (float *)&P[js1]);
    svfloat32x3_t pp2 = svld3_f32(pg2, (float *)&P[js2]);
    svfloat32_t   xj1 = svget3(pp1, 0);
    svfloat32_t   xj2 = svget3(pp2, 0);
    svfloat32_t   yj1 = svget3(pp1, 1);
    svfloat32_t   yj2 = svget3(pp2, 1);
    svint32_t     jj1 = svindex_s32(js1, 1);
    svint32_t     jj2 = svindex_s32(js2, 1);

    svfloat32_t xj = svsplice(pg1, xj1, xj2);
    svfloat32_t yj = svsplice(pg1, yj1, yj2);
    svint32_t   jj = svsplice(pg1, jj1, jj2);

    svbool_t pg =  svwhilelt_b32(0, cnt1+cnt2);

    svfloat32_t dx = svsub_x(pg, xj, xi);
    svfloat32_t dy = svsub_x(pg, yj, yi);
    svfloat32_t r2 = svmul_x(pg, dx, dx);
    r2 = svmla_x(pg, r2, dy, dy);

    svbool_t  cmp   = svcmplt(pg, r2, h2v);
    int       cnt   = svcntp_b32(pg, cmp);
    svint32_t jcomp = svcompact(cmp, jj);
    svbool_t  mask  = svwhilelt_b32(0, cnt);  

    svst1_s32(mask, jbuf+nj, jcomp);
    nj += cnt;
  };

  for(int i=ibeg; i<iend; i++){
    int nj = 0;
    int jbuf[16];

    int cnt1 = cend - (i+1);
    int cnt2 = rend - rbeg;
    if(cnt1 + cnt2 <= 16){
      search2(i, i+1, cend, rbeg, rend, nj, jbuf);
    }else{
      search(i, i+1,  cend, nj, jbuf);
      search(i, rbeg, rend, nj, jbuf);
    }
    assert(nj < 16);

    for(int k=0; k<nj; k++){
      int j = jbuf[k];
      if(P->dist2(P[i], P[j], X, Y) < h2){
	nblist.emplace_back(P[i], P[j], X, Y);
      }
    }
  }
}
#endif

int main(int argc, char **argv){
  int N = 1000'000;
  if(argc > 1) N = atoi(argv[1]);
  sc::input(N);

  enum{
    NTH_MAX =  48,
    MAJOR   = 768,
    MINOR   =  20,
  };
  int major = MAJOR;
  if(argc > 2) major = atoi(argv[2]);
  int minor = MINOR;
  if(argc > 3) minor = atoi(argv[3]);
  int seed = 334;
  if(argc > 4) seed  = atoi(argv[4]);
  double hbase = 27.;
  if(argc > 5) hbase = atof(argv[5]);

  int nbins = major * minor;
  double h = hbase / N;
  printf("h=%e, nbins=%d, n_avr=%f\n", h, nbins, 1.0*N/nbins/nbins);
  assert(h <= 1.0/nbins);

  int nth;
#pragma omp parallel
  nth = omp_get_num_threads();

  assert(nth <= NTH_MAX);

  double t0 = omp_get_wtime();
  Point *P = (Point *)malloc(N * sizeof(Point));
  Offset2D off(major, minor);
  std::vector<Point> pbuf[NTH_MAX];
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    pbuf[tid].reserve((N + N/8) / major);
  }

  puts("initialized");

  double t2 = omp_get_wtime();
  fapp_start("sort_1_1", 0, 0);
  int xh_count[nth+1][major];
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int *cnt = xh_count[tid+1];
    for(int k=0; k<major; k++){
      cnt[k] = 0;
    }
#pragma omp for
    for(int i=0; i<N; i++){
      int ix = off.index_xh(sc::X[i]);
      cnt[ix]++;
    }

#pragma omp for nowait
    for(int k=0; k<major; k++){
      xh_count[0][k] = 0;
    }
    for(int t=0; t<nth; t++){
#pragma omp for nowait
      for(int k=0; k<major; k++){
	xh_count[t+1][k] += xh_count[t][k];
      }
    }
  } // end parallel

  int xh_off[major+1];
  for(int xh=0, sum=0; xh<=major; xh++){
    xh_off[xh] = sum;
    sum += xh_count[nth][xh];
  }
  assert(xh_off[major] == N); // pass

  double t3 = omp_get_wtime();
  fapp_stop ("sort_1_1", 0, 0);
  fapp_start("sort_1_2", 0, 0);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int *th_off = xh_count[tid];
#pragma omp for
    for(int i=0; i<N; i++){
      int ix = off.index_xh(sc::X[i]);
      Point pp(sc::X[i], sc::Y[i], i);
      P[xh_off[ix] + th_off[ix]++] = pp;
    }
  }

  double t4 = omp_get_wtime();
  fapp_stop ("sort_1_2", 0, 0);
  fapp_start("sort_2", 0, 0);
  // sort-2
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto &ptmp = pbuf[tid];

    int xy_size = minor*minor;

#pragma omp for
    for(int xh=0; xh<major; xh++){
      int xy_count[xy_size], xy_off[xy_size+1];
      const int np = xh_off[xh+1] - xh_off[xh];
      Point *psrc = &P[xh_off[xh]];
      assert(np < (int)ptmp.capacity());
      ptmp.resize(np);
      // just count
      for(int k=0; k<xy_size; k++){ xy_count[k] = 0; }
      for(int i=0; i<np; i++){
	int ixy = off.index_xlyh(psrc[i].x, psrc[i].y);
	xy_count[ixy]++;
      }
      // make offsets
      for(int ixy=0, sum=0; ixy<=xy_size; ixy++){
	xy_off[ixy] = sum;
	sum += xy_count[ixy];
      }
      assert(np == xy_off[xy_size]);
      // move
      int xy_loc[xy_size];
      memcpy(xy_loc, xy_off, sizeof(xy_loc));
      for(int i=0; i<np; i++){
	int ixy = off.index_xlyh(psrc[i].x, psrc[i].y);
	ptmp[xy_loc[ixy]++] = psrc[i];
      }
      // final y-sort
      for(int ixy=0; ixy<xy_size; ixy++){
	int beg = xy_off[ixy];
	int end = xy_off[ixy+1];
	int nn = end - beg;
	Point *pdst = psrc  + xy_off[ixy];
	int yl_count[major], yl_off[major+1];
	// just count
	for(int k=0; k<major; k++){ yl_count[k] = 0; }
	for(int i=beg; i<end; i++){
	  int yl = off.index_yl(ptmp[i].y);
	  yl_count[yl]++;
	}
	// make offsets
	for(int yl=0, sum=0; yl<=major; yl++){
	  yl_off[yl] = sum;
	  sum += yl_count[yl];
	}
	assert(nn == yl_off[major]);
	// move
	int yl_loc[major];
	memcpy(yl_loc, yl_off, sizeof(yl_loc));
	for(int i=beg; i<end; i++){
	  int yl = off.index_yl(ptmp[i].y);
	  pdst[yl_loc[yl]++] = ptmp[i];
	}
	// write offset
	for(int yl=0; yl<=major; yl++){
	  off.loc3(xh, ixy, yl) = xh_off[xh] + xy_off[ixy] + yl_off[yl];
	}
      }
    }
  } // end parallel
  assert(N == off.loc(nbins-1, nbins));
  double t5 = omp_get_wtime();
  fapp_stop ("sort_2", 0, 0);
  fapp_start("nbsearch", 0, 0);
  // Count neighbours
  using std::max;
  using std::min;
  using std::vector;
  static int nnb[NTH_MAX];
  static vector<Pair2> nblist[NTH_MAX];
  double h2 = h * h;
  float h2f = h + 2*FLT_EPSILON;
  h2f *= h2f;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    nnb[tid] = 0;
    nblist[tid].reserve(100);
    nblist[tid].clear();
#pragma omp for
    for(int ix=0; ix<nbins; ix++){
      for(int iy=0; iy<nbins; iy++){
	int ub = nbins - 1;
	int ibeg = off.loc(ix, iy);
	int iend = off.loc(ix, iy+1);
	int cend = off.loc(ix, min(iy+1, ub) + 1);
	int rbeg = (ix==ub) ? 0 : off.loc(ix+1, max(iy-1, 0));
	int rend = (ix==ub) ? 0 : off.loc(ix+1, min(iy+1, ub) + 1);

#ifdef __ARM_FEATURE_SVE
	search_nnb3(h2, h2f, ibeg, iend, cend, rbeg, rend, P, sc::X, sc::Y, nblist[tid]);
#else
	search_nnb (h2, h2f, ibeg, iend, cend, rbeg, rend, P, sc::X, sc::Y, nblist[tid]);
#endif
      }
    }
    nnb[tid] = nblist[tid].size();
  }

  int nnbtot = 0;
  vector<Pair2> list_tot;
  list_tot.reserve(2000);
  for(int t=0; t<nth; t++){
    nnbtot += nnb[t];
    list_tot.insert(list_tot.end(), nblist[t].begin(), nblist[t].end());
    assert(nnb[t] == (int)nblist[t].size());
  }
  assert(nnbtot == (int)list_tot.size());
  printf("found %zu\n",  list_tot.size());
  std::sort(list_tot.begin(), list_tot.end());

  double t6 = omp_get_wtime();
  fapp_stop ("nbsearch", 0, 0);
  size_t hash = 0;
  for(int k=0; k<1000; k++){
    hash = list_tot[k].hash(hash);
    sc::pairs[k].i = list_tot[k].i;
    sc::pairs[k].j = list_tot[k].j;
    sc::pairs[k].dist2 = list_tot[k].dist2;
  }
  printf(" hash : %zx\n", hash);
  printf("----------------------\n");
  printf("sort-1,1 : %f\n", t3 - t2);
  printf("sort-1,2 : %f\n", t4 - t3);
  printf("sort-2   : %f\n", t5 - t4);
  printf("nbsearch : %f\n", t6 - t5);
  printf("----------------------\n");
  printf("total    : %f\n", t6 - t2);

  sc::output();
  sc::finalize();
  return 0;
}
