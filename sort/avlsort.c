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

struct node* rotate_left(struct node* root) {
  struct node* right_child = root->right; // right
  root->right = right_child->left; // right
  right_child->left = root;
  return right_child;
}

struct node* rotate_right(struct node* root) {
  struct node* left_child = root->left;
  root->left = left_child->right;
  left_child->right = root;
  return left_child;
}

int max_height = 0;

int height(struct node* root) {
  if (root == NULL) return -1;                
  int left = 1 + height(root->left);
  int right = 1 + height(root->right);
  int height = left > right ? left : right; // ?
  max_height = max_height > height ? max_height : height;
  return height;
}

struct node* insert(struct node* root, int data) {
  if (root == NULL) root = create(data);
  else if (data > root->data)
  {
    root->right = insert(root->right, data);
    if (height(root->left) - height(root->right) == -2) {
      if (data >= root->right->data)
        root = rotate_left(root);
      else {
        root->right = rotate_right(root->right);
        root = rotate_left(root);
      }
    }
  }
  else
  {
    root->left = insert(root->left, data);
    if (height(root->left) - height(root->right) == 2) {
      if (data <= root->left->data)
        root = rotate_right(root);
      else {
        root->left = rotate_left(root->left);
        root = rotate_right(root);
      }
    }
  }
  return root;
}

void traversal(struct node* root) {
  if (root == NULL) return;
  traversal(root->left);
  printf("%d ", root->data);
  traversal(root->right);
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
