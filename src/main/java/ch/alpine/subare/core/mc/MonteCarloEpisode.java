// code by jph
package ch.alpine.subare.core.mc;

import java.util.Queue;

import ch.alpine.subare.core.EpisodeInterface;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.StepInterface;
import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.subare.core.util.PolicyWrap;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** useful to generate a trajectory/episode that is result of taking
 * actions according to a given policy from a given start state */
public final class MonteCarloEpisode implements EpisodeInterface {
  private final MonteCarloInterface monteCarloInterface;
  private final Policy policy;
  private Tensor state;
  private final Queue<Tensor> openingActions;

  /** @param monteCarloInterface
   * @param policy
   * @param state start of episode
   * @param openingActions */
  public MonteCarloEpisode(MonteCarloInterface monteCarloInterface, Policy policy, //
      Tensor state, Queue<Tensor> openingActions) {
    this.monteCarloInterface = monteCarloInterface;
    this.policy = policy;
    this.state = state;
    this.openingActions = openingActions;
  }

  @Override // from EpisodeInterface
  public StepInterface step() {
    final Tensor prev = state;
    final Tensor action;
    if (openingActions.isEmpty()) {
      PolicyWrap policyWrap = new PolicyWrap(policy);
      action = policyWrap.next(state, monteCarloInterface);
    } else {
      action = openingActions.poll();
    }
    final Tensor next = monteCarloInterface.move(state, action);
    final Scalar reward = monteCarloInterface.reward(state, action, next);
    state = next;
    return new StepAdapter(prev, action, reward, next);
  }

  @Override // from EpisodeInterface
  public boolean hasNext() {
    return !monteCarloInterface.isTerminal(state);
  }
}
