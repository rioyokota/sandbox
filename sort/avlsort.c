#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct node {
  int data;
  struct node* left;
  struct node* right;
};

struct node* create(int data) {
  struct node* new_node = (struct node*) malloc (sizeof(struct node));
  new_node->data = data; // data
  new_node->left = NULL;
  new_node->right = NULL;
  return new_node; // new_node
}

struct node* rotate_left(struct node* p) {
  struct node* right_child = p->right; // right
  p->right = right_child->left; // right
  right_child->left = p;
  return right_child;
}

struct node* rotate_right(struct node* p) {
  struct node* left_child = p->left;
  p->left = left_child->right;
  left_child->right = p;
  return left_child;
}

int max_height = 0;

int height(struct node* p) {
  if (p == NULL) return -1;
  int left = 1 + height(p->left);
  int right = 1 + height(p->right);
  int height = left > right ? left : right; // ?
  max_height = max_height > height ? max_height : height;
  return height;
}

struct node* insert(struct node* p, int data) {
  if (p == NULL) p = create(data);
  else if (data > p->data)
  {
    p->right = insert(p->right, data);
    if (height(p->left) - height(p->right) == -2) {
      if (data >= p->right->data)
        p = rotate_left(p);
      else {
        p->right = rotate_right(p->right);
        p = rotate_left(p);
      }
    }
  }
  else
  {
    p->left = insert(p->left, data);
    if (height(p->left) - height(p->right) == 2) {
      if (data <= p->left->data)
        p = rotate_right(p);
      else {
        p->left = rotate_left(p->left);
        p = rotate_right(p);
      }
    }
  }
  return p;
}

void traversal(struct node* p) {
  if (p == NULL) return;
  traversal(p->left); // a)
  printf("%d ", p->data);  // c)
  traversal(p->right); // b)
}

int main() {
  int N = 70;
  struct node* root = NULL;
  srand((unsigned int)time(NULL));
  for (int i=0; i<N; i++) {
    int data = rand() % N;
    root = insert(root, data);
    printf("%d ",data);
  }
  printf("\n");
  traversal(root);
  printf("\n");
  printf("%d\n",max_height);
}
