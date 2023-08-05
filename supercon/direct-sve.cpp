#include <sc_header.hpp>
#include <queue>
#include <arm_sve.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

int main(int argc, char **argv){
  int N = 1024;
  if(argc > 1)  N = atoi(argv[1]);
  sc::input(N);

  std::priority_queue<sc::Pair> queue;
  double r2_max = 2.0;
  for(int i=0; i<N; i++){
    svfloat64_t xi  = svdup_f64(sc::X[i]);
    svfloat64_t yi  = svdup_f64(sc::Y[i]);
    svfloat64_t vr2_max = svdup_f64(r2_max);
    const auto vlen = svcntd();
    double   r2buf[vlen];
    uint64_t jbuf [vlen];
    int j = i+1;
    svbool_t pg = svwhilelt_b64(j, N);
    while(svptest_any(svptrue_b64(), pg)){
      svuint64_t jindeces = svindex_u64(j, 1);
      svfloat64_t xj = svld1(pg, sc::X+j);
      svfloat64_t yj = svld1(pg, sc::Y+j);
      svfloat64_t dx = svsub_x(pg, xj, xi);
      svfloat64_t dy = svsub_x(pg, yj, yi);
      svfloat64_t r2 = svmul_x(pg, dx, dx);
      r2 = svmla_x(pg, r2, dy, dy);
      svbool_t cmp = svcmplt(pg, r2, vr2_max);
      if(unlikely(svptest_any(svptrue_b64(), cmp))){
	svfloat64_t r2c = svcompact(cmp, r2);
	svuint64_t  jc  = svcompact(cmp, jindeces);
	int num = svcntp_b64(svptrue_b64(), cmp);
	svbool_t mask = svwhilelt_b64(0, num);
	svst1(mask, r2buf, r2c);
	svst1(mask, jbuf, jc  );
	for(int k=0; k<num; k++){
	  double r2 = r2buf[k];
	  if(r2 < r2_max){
	    int jj = jbuf[k];
	    if(queue.size() < 1000){
	      queue.push({i, jj, r2});
	    }else if (r2 < queue.top().dist2) {
	      queue.pop();
	      queue.push({i, jj, r2});
	      r2_max = queue.top().dist2;
	      vr2_max = svdup_f64(r2_max);
	    }
	  }
	}
      }
      j += svcntd();
      pg = svwhilelt_b64(j, N);
    }
  }
  for(int i=0; i<1000; i++){
    sc::pairs[999-i] = queue.top();
    queue.pop();
  }

  sc::output();
  sc::finalize();
  return 0;
};
