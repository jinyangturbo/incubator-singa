#include "neuralnet/layer.h"
#include "myproto.pb.h"

namespace singa {
class HiddenLayer : public NeuronLayer {
 public:
  ~HiddenLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }

 private:
  int batchsize_;
  int vdim_, hdim_;
  bool transpose_;
  Param *weight_, *bias_;
};
}
