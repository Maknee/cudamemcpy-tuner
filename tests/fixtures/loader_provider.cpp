// The provider after loader_caller in the global scope must win RTLD_NEXT.
extern "C" __attribute__((visibility("default"))) int mh_loader_probe() { return 7; }
