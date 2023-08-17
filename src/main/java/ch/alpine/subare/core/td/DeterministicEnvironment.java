// code by jph
package ch.alpine.subare.core.td;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.random.RandomGenerator;

import ch.alpine.subare.core.api.StepDigest;
import ch.alpine.subare.core.api.StepRecord;
import ch.alpine.subare.core.util.StateAction;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/** utility class to implement "Model" for deterministic environments
 * in Tabular Dyna-Q p.172 */
/* package */ class DeterministicEnvironment implements StepDigest {
  private static final RandomGenerator RANDOM = new Random();
  // ---
  private final Map<Tensor, StepRecord> map = new HashMap<>();
  private final Tensor keys = Tensors.empty();

  public StepRecord getRandomStep() {
    return map.get(keys.get(RANDOM.nextInt(size())));
  }

  public StepRecord get(Tensor state, Tensor action) {
    return map.get(StateAction.key(state, action));
  }

  @Override
  public void digest(StepRecord stepInterface) {
    Tensor key = StateAction.key(stepInterface);
    register(key, stepInterface);
  }

  private synchronized void register(Tensor key, StepRecord stepInterface) {
    if (!map.containsKey(key)) {
      map.put(key, stepInterface);
      keys.append(key); // after updating the map, for conservative size
    } else {
      // TODO SUBARE can verify that stored step is identical to provided step
    }
  }

  public int size() {
    return keys.length();
  }
}
