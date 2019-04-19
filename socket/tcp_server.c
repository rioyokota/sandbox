#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define PORT 1024
#define SIZE 16

int main() {
  int s, n, l;
  char buf[SIZE];
  struct sockaddr_in servaddr;
  s = socket(AF_INET, SOCK_STREAM, 0);
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = INADDR_ANY;
  bind(s, (struct sockaddr *)&servaddr, sizeof(servaddr));
  listen(s, 3);
  s = accept(s, (struct sockaddr *)&servaddr, &l);
  n = read(s, buf, strlen(buf));
  buf[n] = '\0';
  printf("%s\n",buf);
  return 0;
}
