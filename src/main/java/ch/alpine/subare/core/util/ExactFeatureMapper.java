// code by fluric
package ch.alpine.subare.core.util;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.UnitVector;

/** requires keys constructed by {@link StateAction}
 * 
 * the implementation initializes the features as unit vectors */
public class ExactFeatureMapper implements FeatureMapper, Serializable {
  /** @param monteCarloInterface
   * @return */
  public static FeatureMapper of(MonteCarloInterface monteCarloInterface) {
    return new ExactFeatureMapper(monteCarloInterface);
  }

  // ---
  private final Map<Tensor, Tensor> stateToFeature = new HashMap<>();
  private final int stateActionSize;

  private ExactFeatureMapper(MonteCarloInterface monteCarloInterface) {
    // count the number of possible state-action pairs first
    stateActionSize = monteCarloInterface.states().stream() //
        .filter(state -> !monteCarloInterface.isTerminal(state)) //
        .mapToInt(state -> monteCarloInterface.actions(state).length()) //
        .sum();
    int index = -1;
    for (Tensor state : monteCarloInterface.states())
      for (Tensor action : monteCarloInterface.actions(state))
        stateToFeature.put(StateAction.key(state, action), //
            monteCarloInterface.isTerminal(state) //
                ? Array.zeros(stateActionSize) // TODO SUBARE this vector is probably never used
                : UnitVector.of(stateActionSize, ++index));
  }

  @Override // from FeatureMapper
  public Tensor getFeature(Tensor key) {
    return stateToFeature.get(key);
  }

  @Override // from FeatureMapper
  public int featureSize() {
    return stateActionSize; // one-to-one mapping
  }
}
