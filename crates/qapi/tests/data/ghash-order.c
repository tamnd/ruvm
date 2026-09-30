// SPDX-License-Identifier: GPL-2.0-or-later
//
// Writes ghash-order.txt: build with cc ghash-order.c $(pkg-config --cflags --libs glib-2.0).
// The output records g_str_hash values and the iteration order after each step of a random
// sequence of inserts and removes, for the GHashTable port in src/ghash.rs to replay.

#include <string.h>
#include <glib.h>
#include <stdio.h>
static unsigned long long s = 12345;
static unsigned rnd(unsigned n) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(s >> 33) % n; }
static const char *words[] = {"type","realized","parent_bus","hotplugged","hotpluggable","id","addr","bus","irq","memory","chardev","device","audiodevs","chardevs","objects","backend","machine","unattached","peripheral","peripheral-anon","sysbus","ioport","legacy-iommu","x-migrate","romfile","multifunction","rombar","failover_pair_id","acpi-index","x-pcie-lnksta-dllla","x-pcie-extcap-init","busnr","x-max-bounce-buffer-size","cpu","kvm-type","dump-guest-core","mem-merge","usb","dt-compatible","firmware"};
static void dump(GHashTable *h) {
  GHashTableIter it; gpointer k;
  printf("=");
  g_hash_table_iter_init(&it, h);
  while (g_hash_table_iter_next(&it, &k, NULL)) printf(" %s", (char*)k);
  printf("\n");
}
int main(void) {
  printf("h");
  for (int i = 0; i < 40; i++) printf(" %u", g_str_hash(words[i]));
  printf("\n");
  for (int round = 0; round < 60; round++) {
    GHashTable *h = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    int n = 1 + rnd(round < 30 ? 40 : 400);
    printf("new\n");
    for (int i = 0; i < n; i++) {
      unsigned op = rnd(10);
      char *key = rnd(3) == 0 ? g_strdup(words[rnd(40)]) : g_strdup_printf("%s[%u]", words[rnd(40)], rnd(64));
      if (op < 7) { printf("i %s\n", key); g_hash_table_insert(h, key, NULL); }
      else if (op < 9) { printf("r %s\n", key); g_hash_table_remove(h, key); g_free(key); }
      else {
        GHashTableIter it; gpointer k;
        printf("R %s\n", key);
        g_hash_table_iter_init(&it, h);
        while (g_hash_table_iter_next(&it, &k, NULL)) if (!strcmp(k, key)) { g_hash_table_iter_remove(&it); break; }
        g_free(key);
      }
      if (rnd(8) == 0) dump(h);
    }
    dump(h);
    g_hash_table_unref(h);
  }
  return 0;
}
