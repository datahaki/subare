package ch.alpine.subare.net.err;

import java.io.Serializable;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public abstract class AbstractProbBinaryCrossEntropy implements LossFunction, Serializable {
  @Override
  public NetLoss compute(Tensor activations, Tensor target) {
    return ProbBinaryCrossEntropy.compute((Scalar) logits(), (Scalar) activations, (Scalar) target);
  }

  @Override
  public NetLoss computeBatch(Tensor activations, Tensor target) {
    return ProbBinaryCrossEntropy.computeBatch(logits(), activations, target);
  }

  protected abstract Tensor logits();

  @Override
  public String toString() {
    return ProbBinaryCrossEntropy.SCALAR.toString();
  }
}
