// code by jph
package ch.alpine.subare.td;

import java.util.PriorityQueue;

import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.sca.Sign;

/** Prioritized Sweeping for a deterministic environment
 * 
 * box on p.172 */
public class PrioritizedSweeping implements StepDigest {
  private final Sarsa sarsa;
  private final int n;
  private final Scalar theta;
  private final DeterministicEnvironment deterministicEnvironment = new DeterministicEnvironment();
  private final PriorityQueue<PrioritizedStateAction> priorityQueue = new PriorityQueue<>();
  private final StateOrigins stateOrigins = new StateOrigins();

  /** @param sarsa underlying learning
   * @param n number of replay steps
   * @param theta non-negative threshold */
  public PrioritizedSweeping(Sarsa sarsa, int n, Scalar theta) {
    this.sarsa = sarsa;
    this.n = n;
    this.theta = Sign.requirePositiveOrZero(theta);
  }
  // public void setPolicy(Policy policy) {
  // sarsa.supplyPolicy(() -> policy);
  // }

  // check priority of learning experience
  private void consider(StepRecord stepRecord) {
    Scalar P = sarsa.priority(stepRecord);
    if (Scalars.lessThan(theta, P))
      priorityQueue.add(new PrioritizedStateAction(P, stepRecord));
  }

  @Override
  public void digest(StepRecord stepRecord) {
    deterministicEnvironment.digest(stepRecord);
    stateOrigins.digest(stepRecord);
    consider(stepRecord);
    // ---
    for (int count = 0; count < n && !priorityQueue.isEmpty(); ++count) {
      PrioritizedStateAction head = priorityQueue.poll();
      final StepRecord model = deterministicEnvironment.get(head.state(), head.action());
      sarsa.digest(model);
      for (StepRecord origin : stateOrigins.values(head.state()))
        consider(origin);
    }
  }
}
