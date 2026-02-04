// code by jph
package ch.alpine.subare.td;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

/** utility class to implement "Model" for deterministic environments
 * in Tabular Dyna-Q p.172 */
/* package */ class DeterministicEnvironment implements StepDigest {
  private final Map<Tensor, StepRecord> map = new HashMap<>();
  private final Tensor keys = Tensors.empty();

  public StepRecord getRandomStep() {
    return map.get(keys.get(ThreadLocalRandom.current().nextInt(size())));
  }

  public StepRecord get(Tensor state, Tensor action) {
    return map.get(StateAction.key(state, action));
  }

  @Override
  public void digest(StepRecord stepRecord) {
    Tensor key = StateAction.key(stepRecord);
    register(key, stepRecord);
  }

  private synchronized void register(Tensor key, StepRecord stepRecord) {
    if (!map.containsKey(key)) {
      map.put(key, stepRecord);
      keys.append(key); // after updating the map, for conservative size
    } else {
      // TODO SUBARE can verify that stored step is identical to provided step
    }
  }

  public int size() {
    return keys.length();
  }
}
