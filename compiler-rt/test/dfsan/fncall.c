// RUN: %clang_dfsan %s -o %t && %run %t
// RUN: %clang_dfsan -mllvm -dfsan-args-abi %s -o %t && %run %t
//
// REQUIRES: x86_64-target-arch
//
// Tests that labels are propagated through function calls.

#include <sanitizer/dfsan_interface.h>
#include <assert.h>

int f(int x) {
  int j = 2;
  dfsan_label j_label = dfsan_create_label("j", 0);
  dfsan_set_label(j_label, &j, sizeof(j));
  return x + j;
}

int main(void) {
  int i = 1;
  dfsan_label i_label = dfsan_create_label("i", 0);
  dfsan_set_label(i_label, &i, sizeof(i));

  dfsan_label ij_label = dfsan_get_label(f(i));
  assert(dfsan_has_label(ij_label, i_label));
  assert(dfsan_has_label_with_desc(ij_label, "j"));

  return 0;
}
