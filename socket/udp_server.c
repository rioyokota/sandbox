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
  struct sockaddr_in servaddr, recvaddr;
  s = socket(AF_INET, SOCK_DGRAM, 0);
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = INADDR_ANY;
  bind(s, (struct sockaddr *)&servaddr, sizeof(servaddr));
  n = recvfrom(s, buf, sizeof(buf), MSG_WAITALL, (struct sockaddr *)&recvaddr, &l);
  buf[n] = '\0';
  printf("%s\n", buf);
  return 0;
}
