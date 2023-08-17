// code by jph
package ch.alpine.subare.td;

import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;

/** Tabular Dyna-Q
 * 
 * box on p.172 */
public class TabularDynaQ implements StepDigest {
  private final Sarsa sarsa;
  private final int n;
  private final DeterministicEnvironment deterministicEnvironment = new DeterministicEnvironment();

  /** @param sarsa underlying learning
   * @param n number of replay steps */
  public TabularDynaQ(Sarsa sarsa, int n) {
    this.sarsa = sarsa;
    this.n = n;
  }

  @Override
  public void digest(StepRecord stepInterface) {
    sarsa.digest(stepInterface);
    deterministicEnvironment.digest(stepInterface);
    // replay previously observed steps:
    int min = Math.min(deterministicEnvironment.size(), n);
    for (int count = 0; count < min; ++count)
      sarsa.digest(deterministicEnvironment.getRandomStep());
  }
}
