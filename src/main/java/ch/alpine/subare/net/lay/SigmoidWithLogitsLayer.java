package ch.alpine.subare.net.lay;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.err.BinaryCrossEntropyWithLogits;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

public class SigmoidWithLogitsLayer extends ElementwiseLayer {
  @Override
  public Tensor forward(Tensor x) {
    return (logits = x).maps(LogisticSigmoid.FUNCTION);
  }

  public LossFunction crossEntropyLoss() {
    return new BinaryCrossEntropyWithLogits() {
      @Override
      protected Tensor logits() {
        return logits;
      }
    };
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput; // from binaryCrossEntropy
  }

  @Override
  public String title() {
    return "SigmoidWithLossLayer";
  }
}
