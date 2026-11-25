package ch.alpine.subare.net.los;

import java.io.Serializable;

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
