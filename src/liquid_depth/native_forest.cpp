#include <cstdint>
#include <cstddef>
extern "C" void forest_predict(int samples, int features, int trees, int width,
    const float* x, const std::int32_t* feature, const double* threshold,
    const std::int32_t* left, const std::int32_t* right,
    const std::uint8_t* leaf, const double* value, double* out) {
  for (int row=0;row<samples;++row) {
    double total=0.0;
    for (int tree=0;tree<trees;++tree) {
      const std::size_t offset=std::size_t(tree)*width;
      int node=0;
      while (!leaf[offset+node]) {
        const auto k=offset+node;
        node=(x[std::size_t(row)*features+feature[k]]<=threshold[k])?left[k]:right[k];
      }
      total+=value[offset+node];
    }
    out[row]=total/trees;
  }
}
