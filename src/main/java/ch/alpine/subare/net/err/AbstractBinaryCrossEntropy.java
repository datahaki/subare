package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Tensor;

public abstract class AbstractBinaryCrossEntropy implements LossFunction, Serializable {
  @Override
  public NetLoss compute(Tensor _p, Tensor target) {
    return BinaryCrossEntropy.SCALAR.compute(logits(), target);
  }

  @Override
  public NetLoss computeBatch(Tensor _p, Tensor target) {
    return BinaryCrossEntropy.SCALAR.computeBatch(logits(), target);
  }

  protected abstract Tensor logits();

  @Override
  public String toString() {
    return BinaryCrossEntropy.SCALAR.toString();
  }
}
