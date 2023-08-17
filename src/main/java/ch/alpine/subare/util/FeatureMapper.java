// code by fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Tensor;

public interface FeatureMapper {
  /** @param key for instance {@link StateAction#key(Tensor, Tensor)}, or {@link StateAction#key(StepRecord)}
   * @return a vector with the features as elements, has to be used as dot product,
   * e.g. the {@link ExactFeatureMapper} returns a unit vector for non-terminal keys */
  Tensor getFeature(Tensor key);

  /** @return returns the length of the feature vector, the number of feature elements,
   * e.g. the {@link ExactFeatureMapper} returns the number of all possible non-terminal state-action combinations */
  int featureSize();
}
