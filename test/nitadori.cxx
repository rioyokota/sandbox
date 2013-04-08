#include<cstdio>
#include<cstdlib>
#include<cassert>
#include<cmath>
#include<complex>

// to store complex X_l^m, where 
//   0<=l<=p, -l<=m<=l, 
//   and X_l^{-m} = (-1)^m conj(X_l^m)
template <int p, typename real_t=double, typename cplx_t = std::complex<real_t> >
struct lmbuf{
	enum {
		order  = p,
		length = (p+1)*(p+1), 
	};
	real_t buf[length];

	void clear(){
		for(int n=0; n<length; n++) buf[n] = real_t(0);
	}

	// X_l^{-m} = (-1)^m conj(X_l^m)
	cplx_t get_at(const int l, const int m) const {
		assert(0 <= l && l <= p);
		assert(abs(m) <= l);
		const real_t re = buf[l*(l+1) + abs(m)];
		const real_t im = buf[l*(l+1) - abs(m)];
		if(m == 0) return cplx_t(re, real_t(0));
		if(m > 0)  return cplx_t(re, im);
		if(m < 0){
			if(m&1) return cplx_t(-re, im);
			else    return cplx_t(re, -im);
		}
		return cplx_t();
	}
	void set_at(const int l, const int m, const cplx_t z){
		assert(0 <= l && l <= p);
		assert(0 <= m && m <= l);
		/*     */ buf[l*(l+1) + m] = real(z);
		if(m > 0) buf[l*(l+1) - m] = imag(z);
	}
	void accum_at(const int l, const int m, const cplx_t z){
		assert(0 <= l && l <= p);
		assert(0 <= m && m <= l);
		/*     */ buf[l*(l+1) + m] += real(z);
		if(m > 0) buf[l*(l+1) - m] += imag(z);
	}
	
	static int idx_re(const int l, const int m){
		return l*(l+1) + m;
	}
	static int idx_im(const int l, const int m){
		return l*(l+1) - m;
	}

	void show(FILE *fp = stdout, const char *fmt = "%+f %+fi, ") const {
		for(int l=0; l<=p; l++){
			fprintf(fp, "l=%d: ", l);
			for(int m=0; m<=l; m++){
				const cplx_t z = get_at(l, m);
				fprintf(fp, fmt, real(z), imag(z));
			}
			fprintf(fp, "\n");
		}
	}
	void show_all(FILE *fp = stdout, const char *fmt = "%+f %+fi, ") const {
		for(int l=0; l<=p; l++){
			fprintf(fp, "l=%d: ", l);
			for(int m=-l; m<=l; m++){
				const cplx_t z = get_at(l, m);
				fprintf(fp, fmt, real(z), imag(z));
			}
			fprintf(fp, "\n");
		}
	}
};

#if 1
static inline double factorial(const int i){
	assert(i >= 0);
	return i ? double(i) * factorial(i-1) 
	         : 1.0;
}
static inline double factinv(const int i){
	return 1.0 / factorial(i);
}
#else
static inline long factorial(const long i){
	assert(i >= 0);
	return i ? i * factorial(i-1) 
	         : 1;
}
static inline double factinv(const int i){
	return 1.0 / double(factorial(i));
}
#endif

template <int p, typename real_t=double, typename cplx_t = std::complex<real_t> >
struct Rlm : public lmbuf<p, real_t, cplx_t>
{
	__attribute__((noinline))
	void eval(const double x, const double y, const double z){
		const cplx_t u(-x, -y); // for (-1)^l
		const real_t r2 = x*x + y*y + z*z;
		const real_t zz = -z;   // for (-1)^l

		cplx_t um(1.0, 0.0);
		real_t pmm = 1.0;
		for(int m=0; m<=p; m++){
			real_t p0 = pmm;
			real_t p1 = real_t(2*m+1) * zz * pmm;
			for(int l=m; l<=p; l++){
				real_t plm;
				if(l == m){
					plm = p0;
				}else if(l == m+1){
					plm = p1;
				}else{
					plm = (1.0/real_t(l-m)) * (
							real_t(2*l-1) * zz * p1
						  - real_t(l+m-1) * r2 * p0);
					p0 = p1;
					p1 = plm;
				}
				const cplx_t val = (factinv(l+m) * plm) * um;
				this->set_at(l, m, val);
			}
			// end of for(m), update values
			um  *= u;
			pmm *= real_t(-(2*m+1));
		}
	}
	__attribute__((noinline))
	void eval_opt(const double x, const double y, const double z){
		static bool initcall = true;
		static real_t tbl_inv[p+1];
		static real_t tbl_factinv[2*p+1];
		if(initcall){
			initcall = false;
			for(int i=0; i<p+1; i++){
				tbl_inv[i] = real_t(1.0) / real_t(i);
			}
			for(int i=0; i<2*p+1; i++){
				assert(factorial(i) > 0);
				tbl_factinv[i] = 1.0 / factorial(i);
			}
		}

		const cplx_t u(-x, -y); // for (-1)^l
		const real_t r2 = x*x + y*y + z*z;
		const real_t zz = -z;   // for (-1)^l

		cplx_t um(1.0, 0.0);
		real_t pmm = 1.0;
		real_t _2mp1 = 1.0;
		for(int m=0; m<=p; m++){
			real_t p0 = pmm;
			real_t p1 = _2mp1 * zz * pmm;
			real_t plm;
			for(int l=m; l<=p; ){
				plm = p0;
				const cplx_t val = (tbl_factinv[l+m] * plm) * um;
				this->set_at(l, m, val);
				break;
			}
			for(int l=m+1; l<=p; ){
				plm = p1;
				const cplx_t val = (tbl_factinv[l+m] * plm) * um;
				this->set_at(l, m, val);
				break;
			}
			real_t c0 = _2mp1;
			real_t c1 = c0 + 2.0;
			for(int l=m+2; l<=p; l++ ){
				plm = (tbl_inv[l-m]) * (c1 * zz * p1 - c0 * r2 * p0);
				p0 = p1;
				p1 = plm;
				c0 += 1.0;
				c1 += 2.0;
				const cplx_t val = (tbl_factinv[l+m] * plm) * um;
				this->set_at(l, m, val);
			}
			// end of for(m), update values
			um  *= u;
			pmm *= real_t(-(2*m+1));
			_2mp1 += 2.0;
		}
	}
};

template <int p, typename real_t=double, typename cplx_t = std::complex<real_t> >
struct Slm : public lmbuf<p, real_t, cplx_t>
{
	__attribute__((noinline))
	void eval(const double x, const double y, const double z){
		const cplx_t u(x, y); // for (-1)^(l+m)
		const real_t r2 = x*x + y*y + z*z;
		const real_t zz = -z;   // for (-1)^l

		const real_t rinv2 = 1.0/r2;
		const real_t rinv  = sqrt(rinv2);

		cplx_t um(1.0, 0.0);
		real_t pmm = 1.0;
		real_t rinv_2mp1 = rinv;
		for(int m=0; m<=p; m++){
			real_t p0 = pmm;
			real_t p1 = (2*m+1) * zz * pmm;
			real_t rinv_2lp1 = rinv_2mp1;
			for(int l=m; l<=p; l++){
				real_t plm;
				if(l == m){
					plm = p0;
				}else if(l == m+1){
					plm = p1;
				}else{
					plm = (1.0/real_t(l-m)) * (
							real_t(2*l-1) * zz * p1
						  - real_t(l+m-1) * r2 * p0);
					p0 = p1;
					p1 = plm;
				}
				const cplx_t val = (rinv_2lp1 * factorial(l-m) * plm) * um;
				this->set_at(l, m, val);
				rinv_2lp1 *= rinv2;
			}
			// end of for(m), update values
			um  *= u;
			pmm *= real_t(-(2*m+1));
			rinv_2mp1 *= rinv2;
		}
	}
	__attribute__((noinline))
	void eval_opt(const double x, const double y, const double z){
		static bool initcall = true;
		static real_t tbl_inv[ p+1];
		// static real_t tbl_fact[p+1];
		if(initcall){
			initcall = false;
			for(int i=0; i<p+1; i++){
				tbl_inv [i] = real_t(1.0) / real_t(i);
				// tbl_fact[i] = real_t(factorial(i));
			}
		}

		const cplx_t u(x, y); // for (-1)^(l+m)
		const real_t r2 = x*x + y*y + z*z;
		const real_t zz = -z;   // for (-1)^l

		const real_t rinv2 = 1.0/r2;
		const real_t rinv  = sqrt(rinv2);

		cplx_t um(1.0, 0.0);
		real_t pmm = 1.0;
		real_t rinv_2mp1 = rinv;
		real_t _2mp1 = 1.0;
		for(int m=0; m<=p; m++){
			real_t rinv_2lp1 = rinv_2mp1;

			real_t p0, p1, plm;
			for(int l=m; l<=p; ){
				plm = p0 = pmm;
				const cplx_t val = (rinv_2lp1 * plm) * um;
				this->set_at(l, m, val);
				rinv_2lp1 *= rinv2;
				break;
			}
			for(int l=m+1; l<=p; ){
				plm = p1 = _2mp1 * zz * pmm;
				const cplx_t val = (rinv_2lp1 * plm) * um;
				this->set_at(l, m, val);
				rinv_2lp1 *= rinv2;
				break;
			}
			real_t fact_lmm = 2.0;
			real_t c0 = _2mp1;
			real_t c1 = c0 + 2.0;
			for(int l=m+2; l<=p; l++ ){
				plm = (tbl_inv[l-m]) * (
						c1 * zz * p1
					  - c0 * r2 * p0);
				p0 = p1;
				p1 = plm;
				c0 += 1.0;
				c1 += 2.0;
				const cplx_t val = (rinv_2lp1 * fact_lmm * plm) * um;
				this->set_at(l, m, val);
				rinv_2lp1 *= rinv2;
				fact_lmm *= real_t(l-m+1);
			}
			// end of for(m), update values
			um  *= u;
			pmm *= -_2mp1;
			rinv_2mp1 *= rinv2;
			_2mp1 += 2.0;
		}
	}
};

template <int p, typename real_t=double, typename cplx_t = std::complex<real_t> >
struct MultipoleMoment : public lmbuf<p, real_t, cplx_t> {
	MultipoleMoment(){
		this->clear();
	}
	void assign_particle( // P2M
			const double x, 
			const double y, 
			const double z, 
			const double charge)
	{
		Rlm<p, real_t, cplx_t> rlm;
		rlm.eval_opt(x, y, z);
		for(int l=0; l<=p; l++){
			for(int m=0; m<=l; m++){
				cplx_t val = charge * rlm.get_at(l, -m);
				this->accum_at(l, m, val);
			}
		}
	}
	void assign_particle_opt( // P2M
			const double x, 
			const double y, 
			const double z, 
			const double charge)
	{
		Rlm<p, real_t, cplx_t> rlm;
		rlm.eval_opt(-x, y, z);
		for(int n=0; n<(this->length); n++){
			this->buf[n] += charge * rlm.buf[n];
		}
	}
	cplx_t eval_cplx_potential( // M2P
			const double x, 
			const double y, 
			const double z) 
	{
		Slm<p, real_t, cplx_t> slm;
		slm.eval_opt(x, y, z);

		cplx_t pot(0.0, 0.0);
		for(int l=0; l<=p; l++){
			for(int m=-l; m<=l; m++){
				pot += this->get_at(l,m) * slm.get_at(l,m);
			}
		}
		return pot;
	}
	real_t eval_potential(
			const double x, 
			const double y, 
			const double z) 
	{
		Slm<p, real_t, cplx_t> slm;
		slm.eval_opt(x, y, z);

		real_t pot = 0.0;
		for(int l=0; l<=p; l++){
			pot += real(this->get_at(l, 0) * slm.get_at(l, 0));
			for(int m=1; m<=l; m++){
				pot += 2.0 * real(this->get_at(l,m) * slm.get_at(l,m));
			}
		}
		return pot;
	}
	
};

template <int p, typename real_t=double, typename cplx_t = std::complex<real_t> >
struct LocalExpansion : public lmbuf<p, real_t, cplx_t> {
	LocalExpansion(){
		this->clear();
	}

	void pick_phi_and_grad(real_t &phi, real_t &ex, real_t &ey, real_t &ez) const {
		phi =  this->buf[0]; // (0,0)
		ex  =  this->buf[3]; // Re(1,1)
		ey  = -this->buf[1]; // Im(1,1)
		ez  = -this->buf[2]; // (1,0)
	}

	// L2P
	real_t eval_potential(const real_t x, const real_t y, const real_t z) const {
		Rlm<p, real_t, cplx_t> rlm;
		rlm.eval_opt(x, y, z);

		real_t pot = 0.0;
		for(int l=0; l<=p; l++){
			pot += real(this->get_at(l, 0) * rlm.get_at(l, 0));
			for(int m=1; m<=l; m++){
				pot += 2.0 * real(this->get_at(l,m) * rlm.get_at(l,m));
			}
		}
		return pot;
	}

	// L2L
	template <int psrc>
	void assign_from_LE(
			const LocalExpansion<psrc, real_t, cplx_t> &LEsrc,
			const real_t x,
			const real_t y,
			const real_t z)
	{
		Rlm<psrc, real_t, cplx_t> rlm;
		rlm.eval_opt(-x, -y, -z);

		for(int l=0; l<=p; l++){
			for(int m=0; m<=l; m++){
				cplx_t val(0.0, 0.0);
				for(int lsrc=l; lsrc<=psrc; lsrc++){
#if 0
					for(int msrc=-lsrc; msrc<=lsrc; msrc++){
						if(abs(msrc-m) > (lsrc-l)) continue;
						printf("L'(%d, %d) <= L(%d, %d) *  R(%d, %d)\n", 
						 		l, m, lsrc, msrc, lsrc-l, msrc-m);
						val += LEsrc.get_at(lsrc, msrc) * rlm.get_at(lsrc-l, msrc-m);
					}
#else
					for(int msrc=-(lsrc-l)+m; msrc<=(lsrc-l)+m; msrc++){
						//printf("L'(%d, %d) <= L(%d, %d) *  R(%d, %d)\n", 
						// 		l, m, lsrc, msrc, lsrc-l, msrc-m);
						val += LEsrc.get_at(lsrc, msrc) * rlm.get_at(lsrc-l, msrc-m);
					}
#endif
				}
				this->accum_at(l, m, val);
			}
		}
	}
	// M2L
	template <int psrc>
	void assign_from_MM(
			const MultipoleMoment<psrc, real_t, cplx_t> &MMsrc,
			const real_t x,
			const real_t y,
			const real_t z)
	{
		Slm<p+psrc, real_t, cplx_t> slm;
		slm.eval_opt(x, y, z);

		for(int l=0; l<=p; l++){
			for(int m=0; m<=l; m++){
				cplx_t val(0.0, 0.0);
				for(int lsrc=0; lsrc<=psrc; lsrc++){
					const real_t sign = (lsrc&1) ? (-1.0) : (+1.0);
					for(int msrc=-lsrc; msrc<=lsrc; msrc++){
						val += sign * (MMsrc.get_at(lsrc, msrc) * slm.get_at(lsrc+l, msrc-m));
						//printf("L(%d, %d) <= M(%d, %d) *  L(%d, %d)\n", 
						// 		l, m, lsrc, msrc, lsrc+l, msrc-m);
					}
				}
				this->accum_at(l, m, val);
			}
		}
	}
};

template <int p, typename real_t=double, typename cplx_t = std::complex<real_t> >
struct Matrix{
	typedef lmbuf<p, real_t, cplx_t> lmdef;
	real_t mat[lmdef::length][lmdef::length];

	Slm<p+p, real_t, cplx_t> slm;
	cplx_t elem_M2L(const int lsrc, const int msrc, const int ldst, const int mdst) const {
			const real_t sign = (lsrc&1) ? (-1.0) : (+1.0);
			// printf("(%d, %d) = (%d+%d, %d-%d)\n", lsrc+ldst, msrc-mdst, lsrc, ldst, msrc, mdst);
			return sign * slm.get_at(lsrc+ldst, msrc-mdst);
	}

	void gen_matrix_M2L(const real_t x, const real_t y, const real_t z){
		slm.eval_opt(x, y, z);

		for(int ldst=0; ldst<=p; ldst++){
			for(int mdst=0; mdst<=ldst; mdst++){
				const int row_re = lmdef::idx_re(ldst, mdst);
				const int row_im = lmdef::idx_im(ldst, mdst);
				for(int lsrc=0; lsrc<=p; lsrc++){
					for(int msrc=0; msrc<1; msrc++){
						const int col_re = lmdef::idx_re(lsrc, msrc);
						mat[row_re][col_re] = real(elem_M2L(lsrc, msrc, ldst, mdst));
						if(mdst > 0){
							mat[row_im][col_re] = imag(elem_M2L(lsrc, msrc, ldst, mdst));
						}
					}
					for(int msrc=1; msrc<=lsrc; msrc++){
						const int col_re = lmdef::idx_re(lsrc, msrc);
						const int col_im = lmdef::idx_im(lsrc, msrc);
						const real_t sign = (msrc&1) ? (-1.0) : (+1.0);

						const real_t a = real(elem_M2L(lsrc,  msrc, ldst, mdst));
						const real_t b = real(elem_M2L(lsrc, -msrc, ldst, mdst)) * sign;
						const real_t c = imag(elem_M2L(lsrc,  msrc, ldst, mdst));
						const real_t d = imag(elem_M2L(lsrc, -msrc, ldst, mdst)) * sign;

						mat[row_re][col_re] =  a + b;
						mat[row_re][col_im] = -c + d;
						if(mdst > 0){
							mat[row_im][col_re] = c + d;
							mat[row_im][col_im] = a - b;
						}
					}
				}
			}
		}
	}

	void transpose_with_matrix(
			const MultipoleMoment<p, real_t, cplx_t> &mm,
			      LocalExpansion <p, real_t, cplx_t> &le) const
	{
		for(int i=0; i<lmdef::length; i++){
			real_t sum = 0.0;
			for(int j=0; j<lmdef::length; j++){
				sum += mat[i][j] * mm.buf[j];
			}
			le.buf[i] = sum;
		}
	}
	void transpose_with_slm(
			const MultipoleMoment<p, real_t, cplx_t> &mm,
			      LocalExpansion <p, real_t, cplx_t> &le) const
	{
		for(int ldst=0; ldst<=p; ldst++){
			for(int mdst=0; mdst<=ldst; mdst++){
				cplx_t val(0.0, 0.0);
				for(int lsrc=0; lsrc<=p; lsrc++){
					for(int msrc=-lsrc; msrc<=lsrc; msrc++){
						val += mm.get_at(lsrc, msrc) * elem_M2L(lsrc, msrc, ldst, mdst);
					}
				}
				le.set_at(ldst, mdst, val);
			}
		}
	}
};

#if 0
int main(){
	const char *fmt = "%+10.4e%+10.4ei  ";

	Rlm<5> rlm;
	rlm.eval(1.0, 2.0, 3.0);
	rlm.show(stdout, fmt);
	puts("");
	rlm.eval_opt(1.0, 2.0, 3.0);
	rlm.show(stdout, fmt);
	puts("");

	Slm<5> slm;
	slm.eval(1.0, 2.0, 3.0);
	slm.show(stdout, fmt);
	puts("");
	slm.eval_opt(1.0, 2.0, 3.0);
	slm.show(stdout, fmt);
	puts("");

	return 0; 
}
#else

template <int pmax> void mm_errcheck(
		const int Nsrc,
		const double rsrc[][3],
		const double qsrc[],
		const double rdst[3],
		const double phipp)
{
	mm_errcheck<pmax-1> (Nsrc, rsrc, qsrc, rdst, phipp);

	MultipoleMoment<pmax> mm;
	for(int i=0; i<Nsrc; i++){
		mm.assign_particle_opt(rsrc[i][0], rsrc[i][1], rsrc[i][2], qsrc[i]);
	}
	// std::complex<double> phic = mm.eval_cplx_potential(rdst[0], rdst[1], rdst[2]);
	double phir = mm.eval_potential(rdst[0], rdst[1], rdst[2]);
	// printf("phi<%d> = %e = %+e %+ei\n", pmax, phir, real(phic), imag(phic));
	printf("dphi<%d> = %+e\n", pmax, (phir - phipp)/phipp);
}

template <> void mm_errcheck<0>(
		const int Nsrc,
		const double rsrc[][3],
		const double qsrc[],
		const double rdst[3],
		const double phipp)
{
}

// check for M2P (with grad)
template <int p> void errcheck_phi_grad(
		const int Nsrc,
		const double rsrc[][3],
		const double qsrc[],
		const double rdst[3],
		const double phi0,
		const double ex0,
		const double ey0,
		const double ez0)
{
	MultipoleMoment<p> mm;
	for(int i=0; i<Nsrc; i++){
		mm.assign_particle_opt(rsrc[i][0], rsrc[i][1], rsrc[i][2], qsrc[i]);
	}
	LocalExpansion<1> le;
	le.template assign_from_MM<p>(mm, -rdst[0], -rdst[1], -rdst[2]);
	double phi, ex, ey, ez;
	le.pick_phi_and_grad(phi, ex, ey, ez);
	// printf("phi = %+e, E = (%e, %e, %e)\n", phi, ex, ey, ez);
	const double enorm = sqrt(ex0*ex0 + ey0*ey0 + ez0*ez0);
	printf("<%d> : dphi = %+e, dE = (%+e, %+e, %+e)\n", 
			p, (phi-phi0)/phi0, (ex-ex0)/enorm, (ey-ey0)/enorm, (ez-ez0)/enorm);
}
#endif

#if 0
int main(){
	enum{
		N = 10,
		REP = 4,
	};
	double qsrc[N];
	double rsrc[N][3];
	double rdst[3];

	const double a = 0.2;
	for(int itry=0; itry<REP; itry++){
		for(int i=0; i<N; i++){
			qsrc[i] = drand48();
			for(int k=0; k<3; k++){
				rsrc[i][k] = a * (drand48() - 0.5);
			}
		}
		for(int k=0; k<3; k++){
			rdst[k] = drand48();
		}
		const double norm2 = rdst[0]*rdst[0] + rdst[1]*rdst[1] + rdst[2]*rdst[2];
		for(int k=0; k<3; k++){
			rdst[k] *= (1.0 / sqrt(norm2));
		}

		double phi=0.0, ex=0.0, ey=0.0, ez=0.0;
		for(int i=0; i<N; i++){
			double dx = rsrc[i][0] - rdst[0];
			double dy = rsrc[i][1] - rdst[1];
			double dz = rsrc[i][2] - rdst[2];
			double r2 = dx*dx + dy*dy + dz*dz;
			double ri2 = 1.0 / r2;
			double qri = qsrc[i] * sqrt(ri2);
			phi += qri;
			ex  += dx * (qri * ri2);
			ey  += dy * (qri * ri2);
			ez  += dz * (qri * ri2);
		}

		printf("phi = %+e, E = (%e, %e, %e)\n", phi, ex, ey, ez);
		// mm_errcheck<9> (N, rsrc, qsrc, rdst, phi);

		errcheck_phi_grad<3> (N, rsrc, qsrc, rdst, phi, ex, ey, ez);
		errcheck_phi_grad<5> (N, rsrc, qsrc, rdst, phi, ex, ey, ez);
		errcheck_phi_grad<7> (N, rsrc, qsrc, rdst, phi, ex, ey, ez);
		errcheck_phi_grad<9> (N, rsrc, qsrc, rdst, phi, ex, ey, ez);
		puts("");
	}

	// LocalExpansion<5> le5;
	// LocalExpansion<1> le1;
	// le1.assign_from_LE<5>(le5, 1.0, 1.0, 1.0);

	// MultipoleMoment<5> mm5;
	// le1.assign_from_MM<5>(mm5, 1.0, 1.0, 1.0);
	return 0; 
}
#endif

template <int N>
struct Cell{
	double cx, cy, cz;
	double pos   [N][3];
	double charge[N];
	double ef    [N][3];
	double pot   [N];

	void fill_rand(const double a){
		for(int i=0; i<N; i++){
			pos[i][0] = (2.0 * a) * (drand48() - 0.5);
			pos[i][1] = (2.0 * a) * (drand48() - 0.5);
			pos[i][2] = (2.0 * a) * (drand48() - 0.5);
			charge[i] = drand48();
		}
	}

	void eval_phi_direct_from(const Cell &src){
		const double dcx = src.cx - cx;
		const double dcy = src.cy - cy;
		const double dcz = src.cz - cz;
		for(int i=0; i<N; i++){
			double phi=0.0, ex=0.0, ey=0.0, ez=0.0;
			for(int j=0; j<N; j++){
				const double dx = dcx + src.pos[j][0] - pos[i][0];
				const double dy = dcy + src.pos[j][1] - pos[i][1];
				const double dz = dcz + src.pos[j][2] - pos[i][2];
				double r2 = dx*dx + dy*dy + dz*dz;
				double ri2 = 1.0 / r2;
				double qri = src.charge[j] * sqrt(ri2);
				phi += qri;
				ex  += dx * (qri * ri2);
				ey  += dy * (qri * ri2);
				ez  += dz * (qri * ri2);
			}
			ef[i][0] = ex;
			ef[i][1] = ey;
			ef[i][2] = ez;
			pot[i] = phi;
		}
	}

	template <int p>
	void eval_phi_M2L_from(const Cell &src){
		MultipoleMoment<p> mm;
		for(int i=0; i<N; i++){
			mm.assign_particle_opt(
					src.pos[i][0], src.pos[i][1], src.pos[i][2], src.charge[i]);
		}

		const double dcx = src.cx - cx;
		const double dcy = src.cy - cy;
		const double dcz = src.cz - cz;
		LocalExpansion<p> le;
		le.assign_from_MM(mm, dcx, dcy, dcz);

		for(int i=0; i<N; i++){
			LocalExpansion<1> le1;
			le1.assign_from_LE(le, -pos[i][0], -pos[i][1], -pos[i][2]);
			le1.pick_phi_and_grad(pot[i], ef[i][0], ef[i][1], ef[i][2]);
		}
	}

	void subtract_phi(const Cell &rhs){
		for(int i=0; i<N; i++){
			pot[i]   -= rhs.pot[i];
			ef[i][0] -= rhs.ef[i][0];
			ef[i][1] -= rhs.ef[i][1];
			ef[i][2] -= rhs.ef[i][2];
		}
	}

	void show_phi(){
		for(int i=0; i<N; i++){
			fprintf(stdout, "%d : %+e, (%+e, %+e, %+e)\n",
					i, pot[i], ef[i][0], ef[i][1], ef[i][2]);
		}
	}
};

#if 1
int main(){
	Cell<10> csrc, cdst;
	const double a = 0.16;
	csrc.fill_rand(a);
	cdst.fill_rand(a);

	double dx = drand48();
	double dy = drand48();
	double dz = drand48();
	double norm = sqrt(dx*dx + dy*dy + dz*dz);
	csrc.cx = 0.0;
	csrc.cy = 0.0;
	csrc.cz = 0.0;
	cdst.cx = dx / norm;
	cdst.cy = dy / norm;
	cdst.cz = dz / norm;

	cdst.eval_phi_direct_from(csrc);
	cdst.show_phi();
	puts("");

	Cell<10> cdst3(cdst);
	cdst3.eval_phi_M2L_from<3> (csrc);
	cdst3.subtract_phi(cdst);
	cdst3.show_phi();
	puts("");

	Cell<10> cdst5(cdst);
	cdst5.eval_phi_M2L_from<5> (csrc);
	cdst5.subtract_phi(cdst);
	cdst5.show_phi();
	puts("");

	Cell<10> cdst7(cdst);
	cdst7.eval_phi_M2L_from<7> (csrc);
	cdst7.subtract_phi(cdst);
	cdst7.show_phi();
	puts("");

	return 0;
}
#endif
#if 0
int main(){
	enum{
		N = 10,
		p = 5,
	};
	Cell<N> csrc;
	const double a = 0.15;
	csrc.fill_rand(a);

	double dx = drand48();
	double dy = drand48();
	double dz = drand48();
	double norm = sqrt(dx*dx + dy*dy + dz*dz);
	dx /= norm;
	dy /= norm;
	dz /= norm;

	MultipoleMoment<p> mm;
	LocalExpansion <p> le;

	for(int i=0; i<N; i++){
		mm.assign_particle_opt(
				csrc.pos[i][0], csrc.pos[i][1], csrc.pos[i][2], csrc.charge[i]);
	}
	le.assign_from_MM(mm, dx, dy, dz);
	puts("org");
	le.show();
	
	Matrix<p> mat;
	mat.gen_matrix_M2L(dx, dy, dz);
	mat.transpose_with_slm(mm, le);
	puts("slm");
	le.show();
	mat.transpose_with_matrix(mm, le);
	puts("matrix");
	le.show();

	return 0;
}
#endif
