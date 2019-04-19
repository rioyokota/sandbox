#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 1024

int main(int argc, char **argv)  {
  int s;
  struct sockaddr_in servaddr;
  char *buf = argv[1];
  s = socket(AF_INET, SOCK_STREAM, 0);
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  inet_pton(AF_INET, "127.0.0.1", &servaddr.sin_addr);
  connect(s, (struct sockaddr *)&servaddr, sizeof(servaddr));
  send(s, buf, strlen(buf), 0);
  return 0;
}
