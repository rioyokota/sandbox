#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define PORT 1024

int main(int argc, char** argv) {
  int s, n;
  char *buf = argv[1];
  struct sockaddr_in servaddr;
  s = socket(AF_INET, SOCK_DGRAM, 0);
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = INADDR_ANY;
  sendto(s, buf, strlen(buf), MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
  close(s);
  return 0;
}
