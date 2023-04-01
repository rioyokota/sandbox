#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct node {
  int data;
  struct node* left;
  struct node* right;
  int ht;
};

struct node* root = NULL;

struct node* create(int);
struct node* insert(struct node*, int);
struct node* rotate_left(struct node*);
struct node* rotate_right(struct node*);
int balance_factor(struct node*);
int height(struct node*);
void inorder(struct node*);
void preorder(struct node*);
void postorder(struct node*);

struct node* create(int data) {
  struct node* new_node = (struct node*) malloc (sizeof(struct node));
  new_node->data = data;
  new_node->left = NULL;
  new_node->right = NULL;
  return new_node;
}

struct node* rotate_left(struct node* root) {
  struct node* right_child = root->right;
  root->right = right_child->left;
  right_child->left = root;
  root->ht = height(root);
  right_child->ht = height(right_child);
  return right_child;
}

struct node* rotate_right(struct node* root) {
  struct node* left_child = root->left;
  root->left = left_child->right;
  left_child->right = root;
  root->ht = height(root);
  left_child->ht = height(left_child);
  return left_child;
}

int balance_factor(struct node* root) {
  int lh, rh;
  if (root == NULL)
    return 0;
  if (root->left == NULL)
    lh = 0;
  else
    lh = 1 + root->left->ht;
  if (root->right == NULL)
    rh = 0;
  else
    rh = 1 + root->right->ht;
  return lh - rh;
}

int height(struct node* root) {
  int lh, rh;
  if (root == NULL)
    return 0;
  if (root->left == NULL)
    lh = 0;
  else
    lh = 1 + root->left->ht;
  if (root->right == NULL)
    rh = 0;
  else
    rh = 1 + root->right->ht;
  if (lh > rh)
    return (lh);
  return (rh);
}

struct node* insert(struct node* root, int data) {
  if (root == NULL) {
    struct node* new_node = create(data);
    root = new_node;
  }
  else if (data > root->data) {
    root->right = insert(root->right, data);
    if (balance_factor(root) == -2) {
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
    if (balance_factor(root) == 2) {
      if (data <= root->left->data)
        root = rotate_right(root);
      else {
        root->left = rotate_left(root->left);
        root = rotate_right(root);
      }
    }
  }
  root->ht = height(root);
  return root;
}

void inorder(struct node* root) {
  if (root == NULL) return;
  inorder(root->left);
  printf("%d ", root->data);
  inorder(root->right);
}

void preorder(struct node* root) {
  if (root == NULL) return;
  printf("%d ", root->data);
  preorder(root->left);
  preorder(root->right);
}

void postorder(struct node* root) {
  if (root == NULL) return;
  postorder(root->left);
  postorder(root->right);
  printf("%d ", root->data);
}

int main() {
  int N = 50;
  int data[N];
  srand((unsigned int)time(NULL));
  for (int i=0; i<N; i++) {
    data[i] = rand() % N;
    root = insert(root, data[i]);
    printf("%d ",data[i]);
  }
  printf("\n");
  inorder(root);
  printf("\n");
}
