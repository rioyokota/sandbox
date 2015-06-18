#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Body {
  double x, y;
  int group;
};
typedef std::vector<Body> Bodies;
typedef Body* B_iter;

B_iter gen_xy(int count, double radius) {
  B_iter pt = (B_iter) malloc(sizeof(Body) * count);
  for (B_iter p=pt; p<pt+count; p++) {
    double ang = 2 * M_PI * drand48();
    double r = radius * drand48();
    p->x = r * cos(ang);
    p->y = r * sin(ang);
  }
  return pt;
}

inline double dist2(B_iter a, B_iter b) {
  double x = a->x - b->x, y = a->y - b->y;
  return x*x + y*y;
}

inline int nearest(B_iter pt, B_iter cent, int n_cluster, double *d2)
{
  int i, min_i;
  B_iter c;
  double d, min_d;

  min_d = HUGE_VAL;
  min_i = pt->group;
  for (c = cent, i = 0; i < n_cluster; i++, c++) {
    if (min_d > (d = dist2(c, pt))) {
      min_d = d; min_i = i;
    }
  }
  if (d2) *d2 = min_d;
  return min_i;
}

void kpp(B_iter pts, int len, B_iter cent, int n_cent)
{
  int i, j;
  int n_cluster;
  double sum, *d = (double*) malloc(sizeof(double) * len);

  B_iter p, c;
  cent[0] = pts[ rand() % len ];
  for (n_cluster = 1; n_cluster < n_cent; n_cluster++) {
    sum = 0;
    for (j = 0, p = pts; j < len; j++, p++) {
      nearest(p, cent, n_cluster, d + j);
      sum += d[j];
    }
    sum *= drand48();
    for (j = 0, p = pts; j < len; j++, p++) {
      if ((sum -= d[j]) > 0) continue;
      cent[n_cluster] = pts[j];
      break;
    }
  }
  for (j = 0, p = pts; j < len; j++, p++) p->group = nearest(p, cent, n_cluster, 0);
  free(d);
}

B_iter lloyd(B_iter pts, int len, int n_cluster)
{
  int i, j, min_i;
  int changed;

  B_iter cent = (B_iter) malloc(sizeof(Body) * n_cluster), p, c;

  /* assign init grouping randomly */
  //for_len p->group = j % n_cluster;

  /* or call k++ init */
  kpp(pts, len, cent, n_cluster);

  do {
    /* group element for centroids are used as counters */
    for (c = cent, i = 0; i < n_cluster; i++, c++) { c->group = 0; c->x = c->y = 0; }
    for (j = 0, p = pts; j < len; j++, p++) {
      c = cent + p->group;
      c->group++;
      c->x += p->x; c->y += p->y;
    }
    for (c = cent, i = 0; i < n_cluster; i++, c++) { c->x /= c->group; c->y /= c->group; }

    changed = 0;
    /* find closest centroid of each B_iter */
    for (j = 0, p = pts; j < len; j++, p++) {
      min_i = nearest(p, cent, n_cluster, 0);
      if (min_i != p->group) {
	changed++;
	p->group = min_i;
      }
    }
  } while (changed > (len >> 10)); /* stop when 99.9% of B_iters are good */

  for (c = cent, i = 0; i < n_cluster; i++, c++) { c->group = i; }

  return cent;
}

void print_eps(B_iter pts, int len, B_iter cent, int n_cluster)
{
  int i, j;
  B_iter p;
  FILE * fid = fopen("kmeans.dat","w");
  for (B_iter c = cent; c < cent+n_cluster; c++) {
    fprintf(fid, "%d %g %g\n", c-cent, c->x, c->y);
    for (j = 0, p = pts; j < len; j++, p++) {
      if (p->group != c-cent) continue;
      fprintf(fid, "%d %g %g\n", c-cent, p->x, p->y);
    }
  }
  fclose(fid);
}

#define PTS 100000
#define K 14
int main()
{
  int i;
  B_iter v = gen_xy(PTS, 10);
  B_iter c = lloyd(v, PTS, K);
  print_eps(v, PTS, c, K);
  free(v); free(c);
  return 0;
}
