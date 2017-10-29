#include <mpi.h>
#include <bits/stdc++.h>
#include <sys/time.h>
using namespace std;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}
int MPIRANK, MPISIZE;

class DistributionNode;
class Distribution;
class SampleNode;
class Sample;
class Evaluator;
class UnionFind;
class Data;
class Input;

class Data
{
public:
  Data();
  ~Data();
  Sample *s;
  vector<vector<int>> edge;
  vector<vector<int>> cicle;
  int n;
  int score;
  Evaluator *e;
  Data(Sample& s,int n,Evaluator& e);
  void makeData();
};

class DistributionNode
{
public:
  double a;
  double k;
  DistributionNode(){}
  DistributionNode(double a,double k){
    this->a = a;
    this->k = k;
  }
  void learning(){
    a++;
  }
  void reflesh(){
    a = 1;
  }
  ~DistributionNode(){}

};

class Distribution
{
public:
  vector<vector<DistributionNode>> distributions;
  int n;
  double k;
  Distribution(int n,double k){
    this->n = n;
    this->k = k;
    distributions = vector<vector<DistributionNode>>(
                                                     n,
                                                     vector<DistributionNode>(
                                                                              n,
                                                                              DistributionNode(k/2,k)
                                                                              )
                                                     );

  }
  void learning(Data* d){
    for(int i = 0;i < n;i++){
      if(d->cicle[i][0] > i) distributions[i][d->cicle[i][0]].learning();
      if(d->cicle[i][1] > i) distributions[i][d->cicle[i][1]].learning();
    }
  }
  void reflesh(){
    for(int i = 0;i < n;i++){
      for(int j = i+1;j < n;j++){
        distributions[i][j].reflesh();
      }
    }
  }
  ~Distribution(){}
  Distribution(){}

};

default_random_engine engine;
normal_distribution<double> dist;

class SampleNode
{
public:
  double x;
  DistributionNode *d;

  SampleNode(){}
  SampleNode(DistributionNode& d){
    this->d = &d;

  }
  void sampling(){
    double mu = d->a/d->k;
    double sigma = d->a*(d->k-d->a)/d->k/d->k;
    this->x = dist(engine)*sqrt(sigma)+mu;
  }
  ~SampleNode(){}

};

class Sample
{
public:
  Distribution d;
  vector<vector<SampleNode>> sampleNodes;
  int n;
  Sample(){}
  ~Sample(){}
  Sample(Distribution& d,int n){
    this->d = d;
    this->n = n;
    sampleNodes = vector<vector<SampleNode>>(
                                             n,
                                             vector<SampleNode>(
                                                                n
                                                                )
                                             );
    for(int i = 0;i < n;i++){
      for(int j = i+1;j < n;j++){
        sampleNodes[i][j] = SampleNode(d.distributions[i][j]);
      }
    }
  }
  void sampling(){
    for(int i = 0;i < n;i++){
      for(int j = i+1;j < n;j++){
        sampleNodes[i][j].sampling();
      }
    }
  }
  double get(int i,int j){
    return this->sampleNodes[i][j].x;
  }
};

class Evaluator
{
public:
  Evaluator(){}
  ~Evaluator(){}
  vector<vector<int>> distance;
  vector<double> *x;
  vector<double> *y;
  int n;
  Evaluator(int n,vector<double>& x,vector<double> y){
    this->x = &x;
    this->y = &y;
    this->n = n;
    distance = vector<vector<int>>(n,vector<int>(n));
    for(int i = 0;i < n;i++){
      for(int j = 0;j < n;j++){
        distance[i][j] = calc(i,j);
      }
    }
  }
  int evaluate(vector<vector<int>>& cicle){
    int ret = 0;
    for(int i = 0;i < n;i++){
      ret += distance[i][cicle[i][0]];
      ret += distance[i][cicle[i][1]];
    }
    return ret/2;
  }
private:
  int calc(int i,int j){
    double xd,yd;
    xd = (*x)[i] - (*x)[j];
    yd = (*y)[i] - (*y)[j];
    return (int)(sqrt(xd*xd+yd*yd)+0.5);
  }

};

class UnionFind
{
public:
  UnionFind(){}
  ~UnionFind(){}
  vector<int> par;
  vector<int> rnk;
  UnionFind(int n){
    par = vector<int>(n);
    rnk = vector<int>(n);
    for(int i = 0;i < n;i++){
      par[i] = i;
    }
  }
  int find(int x){
    if(par[x] == x) return x;
    else return par[x] = find(par[x]);
  }
  void unite(int x,int y){
    x = find(x);
    y = find(y);
    if(x == y) return;
    if(rnk[x] < rnk[y]) par[x] = y;
    else{
      par[y] = x;
      if(rnk[x] == rnk[y]) rnk[x]++;
    }
  }
  bool same(int x,int y){
    return find(x) == find(y);
  }
};

Data::Data(){}
Data::~Data(){}
Data::Data(Sample& s,int n,Evaluator& e){
  this->s = &s;
  this->n = n;
  this->e = &e;
  cicle = vector<vector<int>>(n,vector<int>(2));
}
void Data::makeData(){
  edge = vector<vector<int>>(n*(n-1)/2,vector<int>(2));
  for(int i=0,k=0;i<n;i++){
    for(int j=i+1;j<n;j++,k++){
      edge[k][0] = i;
      edge[k][1] = j;
    }
  }
  s->sampling();
  sort(edge.begin(),edge.end(),[s=&(this->s)](vector<int> a,vector<int> b){
      return (*s)->get(a[0],a[1]) > (*s)->get(b[0],b[1]);
    });
  UnionFind uf(n);
  vector<int> used = vector<int>(n);
  for(int i = 0,j = 0;i < n-1;i++,j++){
    while(uf.same(edge[j][0],edge[j][1]) || used[edge[j][0]] == 2 || used[edge[j][1]] == 2) j++;
    cicle[edge[j][0]][used[edge[j][0]]] = edge[j][1];
    cicle[edge[j][1]][used[edge[j][1]]] = edge[j][0];
    used[edge[j][0]]++;
    used[edge[j][1]]++;
    uf.unite(edge[j][0],edge[j][1]);
  }
  int start,end;
  for(start = 0;start < n;start++){
    if(used[start] == 1) break;
  }
  for(end = start+1;end < n;end++){
    if(used[end] == 1) break;
  }
  cicle[start][1] = end;
  cicle[end][1] = start;
  score = e->evaluate(cicle);
}

class Input
{
public:
  Input(){}
  ~Input(){}
  string file;
  int n;
  vector<double> x;
  vector<double> y;
  Input(string file){
    this->file = file;
  }
  void run(){
    ifstream in(file);
    cin.rdbuf(in.rdbuf());
    cin >> n;
    x = vector<double>(n);
    y = vector<double>(n);
    for(int i = 0;i < n;i++){
      cin >> x[i] >> y[i];
    }
  }
  vector<double>& getX(){
    return this->x;
  }
  vector<double>& getY(){
    return this->y;
  }
  int getN(){
    return this->n;
  }
};

class Checker
{
public:
  Checker(){}
  ~Checker(){}
  int pre;
  int count;
  const int border = 5;
  vector<int> answer;
  int n;
  Checker(int n){
    pre = -1;
    count = 0;
    answer = vector<int>(n);
    this->n = n;
  }
  void check(Data& d){
    if(d.score == pre) count++;
    else {
      pre = d.score;
      count = 0;
    }
    if(count == border){
      vector<bool> used(n);
      int p = 0;
      for(int i = 0;i < n;i++){
        answer[i] = p;
        used[p] = true;
        if(!used[d.cicle[p][0]]) p = d.cicle[p][0];
        else p = d.cicle[p][1];
        if(!MPIRANK) cout << answer[i] << " ";
      }
      if(!MPIRANK) cout << endl;
      exit(0);
    }
  }

};

int lim = 100;

int main(int argc, char ** argv){
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRANK);
  MPI_Comm_size(MPI_COMM_WORLD, &MPISIZE);
  int k = 1000;
  random_device seed_gen;
  engine = default_random_engine(seed_gen());
  dist = normal_distribution<double> (0,1);
  Input in("rand20.in");
  in.run();
  Evaluator evaluator(in.getN(),in.getX(),in.getY());
  Distribution distribution(in.getN(),k);
  Sample sample(distribution,in.getN());
  vector<Data*> data = vector<Data*>(2*k);
  for(int i = 0;i < 2*k;i++){
    data[i] = new Data(sample,in.getN(),evaluator);
  }
  int local_2k = 2*k/MPISIZE;
  vector<int> sendcicle(2*in.getN()*local_2k);
  vector<int> recvcicle(2*in.getN()*2*k);
  vector<int> sendscore(local_2k);
  vector<int> recvscore(2*k);
  Checker checker(in.getN());
  if(!MPIRANK) cout << "pre ok" << endl;
  while(true){
    double tic = get_time();
    for(int i = 0,ic = 0;i < local_2k;i++){
      data[i]->makeData();
      for(int j = 0;j < in.getN();j++){
        sendcicle[ic++] = data[i]->cicle[j][0];
        sendcicle[ic++] = data[i]->cicle[j][1];
      }
      sendscore[i] = data[i]->score;
    }
    double toc = get_time();
    if(!MPIRANK) printf("makeData: %lf s\n",toc-tic);
    MPI_Allgather(&sendcicle[0], 2*in.getN()*local_2k, MPI_INT,
                  &recvcicle[0], 2*in.getN()*local_2k, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&sendscore[0], local_2k, MPI_INT,
                  &recvscore[0], local_2k, MPI_INT, MPI_COMM_WORLD);
    for(int i = 0,ic=0;i < 2*k;i++){
      for(int j = 0;j < in.getN();j++){
        data[i]->cicle[j][0] = recvcicle[ic++];
        data[i]->cicle[j][1] = recvcicle[ic++];
      }
      data[i]->score = recvscore[i];
    }
    tic = get_time();
    if(!MPIRANK) printf("sendData: %lf s\n",tic-toc);
    sort(data.begin(), data.end(),[](Data* a,Data* b){
        return a->score < b->score;
      });
    toc = get_time();
    if(!MPIRANK) printf("sort    : %lf s\n",toc-tic);
    if(!MPIRANK) cout << data[0]->score << endl;

    checker.check(*data[0]);
    tic = get_time();
    if(!MPIRANK) printf("check   : %lf s\n",tic-toc);

    distribution.reflesh();
    toc = get_time();
    if(!MPIRANK) printf("refresh : %lf s\n",toc-tic);
    for(int i = 0;i < k;i++){
      distribution.learning(data[i]);
    }
    tic = get_time();
    if(!MPIRANK) printf("learning: %lf s\n",tic-toc);
  }
  return 0;
}
