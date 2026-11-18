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
  public Scalar penalty(boolean weightDecay, Tensor weights) {
    return l1Regularization.penalty(weightDecay, weights).add(l2Regularization.penalty(weightDecay, weights));
  }

  @Override
  public Tensor gradient(boolean weightDecay, Tensor weights) {
    return l1Regularization.gradient(weightDecay, weights).add(l2Regularization.gradient(weightDecay, weights));
  }
}
