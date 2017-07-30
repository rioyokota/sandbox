#include <petscksp.h>

double kernel(int i, int j, double * xi, double * yi, double * zi) {
  double dx = xi[i] - xi[j];
  double dy = yi[i] - yi[j];
  double dz = zi[i] - zi[j];
  double r2 = dx * dx + dy * dy + dz * dz;
  return expf(-r2*10000);
  //return sqrtf(1+r2*10000);
  //return 1/sqrtf(1+r2*10000);
}

int main(int argc,char **args) {
  Vec            x, b, u;
  Mat            A;
  KSP            ksp;
  PC             pc;
  PetscReal      norm,tol=1.e-14;
  PetscErrorCode ierr;
  PetscInt       i,j,its,nnz,n=100;
  PetscMPIInt    size;
  PetscBool      nonzeroguess = PETSC_FALSE;
  FILE           *fid;

  const char help[] = "Solves a tridiagonal linear system with KSP.\n\n";
  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only!");
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  double * xi = new double [n];
  double * yi = new double [n];
  double * zi = new double [n];
  nnz = 0;
  for (i=0; i<n; i++) {
    xi[i] = drand48();
    yi[i] = drand48();
    zi[i] = drand48();
  }
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      PetscReal f = kernel(i, j, xi, yi, zi);
      if (fabs(f) > tol) {
        ierr = MatSetValues(A,1,&i,1,&j,&f,INSERT_VALUES);CHKERRQ(ierr);
        nnz++;
      }
    }
  }
  printf("%d/%d\n",nnz,n*n);
  fid = fopen("A.mtx","w");
  fprintf(fid, "%d %d %d\n", n, n, nnz);
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      PetscReal f = kernel(i, j, xi, yi, zi);
      if (fabs(f) > tol) {
        fprintf(fid, "%d %d %lf\n", i+1, j+1, f);
      }
    }
  }
  fclose(fid);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecSet(u,1.0);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
