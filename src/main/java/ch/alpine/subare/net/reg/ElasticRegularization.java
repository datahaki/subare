// code by jph
package ch.alpine.subare.net.reg;

import java.io.Serializable;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

class ElasticRegularization implements NetRegularization, Serializable {
  private final L1Regularization l1Regularization;
  private final L2Regularization l2Regularization;

  public ElasticRegularization(Scalar l1, Scalar l2) {
    l1Regularization = new L1Regularization(l1);
    l2Regularization = new L2Regularization(l2);
  }

  @Override
  public Scalar penalty(boolean decayInLoss, Tensor weights) {
    return l1Regularization.penalty(decayInLoss, weights).add(l2Regularization.penalty(decayInLoss, weights));
  }

  @Override
  public Tensor gradient(boolean decayInLoss, Tensor weights) {
    return l1Regularization.gradient(decayInLoss, weights).add(l2Regularization.gradient(decayInLoss, weights));
  }

  @Override
  public Tensor sneakDecay(boolean decayInLoss, Tensor weights) {
    return l1Regularization.sneakDecay(decayInLoss, weights).add(l2Regularization.sneakDecay(decayInLoss, weights));
  }
}
