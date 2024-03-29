// code by fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** useful to get Qsa values without the need to create all the state-action pair values beforehand
 * 
 * implementation does not support {@link #assign(Tensor, Tensor, Scalar)} */
public class FeatureQsaAdapter implements QsaInterface {
  private final Tensor w;
  private final FeatureMapper featureMapper;

  public FeatureQsaAdapter(Tensor w, FeatureMapper featureMapper) {
    this.w = w;
    this.featureMapper = featureMapper;
  }

  @Override // from QsaInterface
  public Scalar value(Tensor state, Tensor action) {
    Tensor stateAction = StateAction.key(state, action);
    return (Scalar) w.dot(featureMapper.getFeature(stateAction));
  }

  @Override // from QsaInterface
  public void assign(Tensor state, Tensor action, Scalar value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void blend(Tensor state, Tensor action, Scalar value, Scalar alpha) {
    throw new UnsupportedOperationException();
  }

  @Override // from QsaInterface
  public QsaInterface copy() {
    return this;
  }
}
