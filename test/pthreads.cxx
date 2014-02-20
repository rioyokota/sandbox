#include <iostream>
#include <unistd.h>

pthread_mutex_t lock;
int shared_data;

void *thread_function(void *arg) {
  int *tid = new int;
  *tid = *((int*)arg) * 2;
  for( int i=0; i<100000; ++i ) {
    pthread_mutex_lock(&lock);
    shared_data++;
    pthread_mutex_unlock(&lock);
  }
  return (void*)(tid);
}

int main() {
  const int numThreads = 12;
  pthread_t threads[numThreads];
  int thread_args[numThreads];
  pthread_mutex_init(&lock,NULL);
  for( int i=0; i<numThreads; ++i ) {
    thread_args[i] = i;
    pthread_create(&threads[i],NULL,thread_function,(void*)&thread_args[i]);
  }
  for( int i=0; i<10; ++i ) {
    usleep(10);
    std::cout << shared_data << std::endl;
  }
  void *exit_status;
  for( int i=0; i<numThreads; ++i ) {
    pthread_join(threads[i],&exit_status);
    std::cout << *((int*)exit_status) << std::endl;
  }
}
