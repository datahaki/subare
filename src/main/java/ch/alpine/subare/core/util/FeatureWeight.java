// code by fluric
package ch.alpine.subare.core.util;

import java.io.Serializable;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.ext.Integers;

public class FeatureWeight implements Serializable {
  private final FeatureMapper featureMapper;
  /** vector */
  private Tensor w;

  public FeatureWeight(FeatureMapper featureMapper) {
    this.featureMapper = featureMapper;
    w = Array.zeros(featureMapper.featureSize());
  }

  public Tensor get() {
    return w;
  }

  /** @param w vector of same length as feature size */
  public void set(Tensor w) {
    Integers.requireEquals(w.length(), featureMapper.featureSize());
    this.w = w;
  }
}
