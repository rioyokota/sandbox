void foo();
void bar();

int a = 10;
int b = 10;

int main() {
  for (int i=0; i<4; i++) {
    foo();
    bar();
  }
}
