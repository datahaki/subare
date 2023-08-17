// code by fluric
package ch.alpine.subare.td;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.PolicyExt;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.util.FeatureQsaAdapter;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.StateAction;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;

/** implementation of box "True Online Sarsa(lambda) for estimating w'x approx. q_pi or q_*
 * 
 * the evaluation types {@link Expected Reward} and {@link Q-Learning} are adapted from the original Sarsa
 * 
 * https://github.com/idsc-frazzoli/subare/files/2257720/trueOnlineSarsa.pdf
 * 
 * in Section 12.8, p.309 */
public class TrueOnlineSarsa extends AbstractTrueOnlineSarsa {
  private final PolicyExt policy;
  /** feature weight vector w is a long-term memory, accumulating over the lifetime of the system */
  private final FeatureWeight w;
  private final StateActionCounter sac;
  private final int featureSize;
  // ---
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;
  private Scalar nextQOld;

  public TrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, //
      Scalar lambda, FeatureMapper featureMapper, LearningRate learningRate, //
      FeatureWeight w, StateActionCounter sac, PolicyExt policy) {
    super(monteCarloInterface, evaluationType, lambda, learningRate, featureMapper);
    this.sac = sac;
    this.w = w;
    this.policy = policy;
    featureSize = featureMapper.featureSize();
    resetEligibility();
  }

  /** faster when only part of the qsa is required */
  @Override // from QsaInterfaceSupplier
  public final QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(w.get(), featureMapper);
  }

  /** @return unmodifiable weight vector w */
  @Override
  public final Tensor getW() {
    return w.get().unmodifiable();
  }

  @Override // from StepDigest
  public final void digest(StepRecord stepRecord) {
    ((PolicyBase) policy).setQsa(qsaInterface());
    Tensor prevState = stepRecord.prevState();
    Tensor prevAction = stepRecord.action();
    Tensor nextState = stepRecord.nextState();
    Scalar reward = stepRecord.reward();
    // ---
    Scalar alpha = learningRate.alpha(stepRecord, sac);
    Scalar alpha_gamma_lambda = alpha.multiply(gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = (Scalar) w.get().dot(x);
    Scalar nextQ = evaluationType.evaluate(nextState, policy);
    Scalar delta = reward.add(gamma.multiply(nextQ)).subtract(prevQ);
    // eq (12.11)
    z = z.multiply(gamma_lambda) //
        .add(x.multiply(RealScalar.ONE.subtract(z.dot(x).multiply(alpha_gamma_lambda))));
    // ---
    Scalar diffQ = prevQ.subtract(nextQOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(diffQ)));
    Tensor scalex = x.multiply(alpha.multiply(diffQ));
    w.set(w.get().add(scalez).subtract(scalex));
    nextQOld = nextQ;
    // ---
    sac.digest(stepRecord);
    // ---
    if (monteCarloInterface.isTerminal(nextState))
      resetEligibility();
  }

  private void resetEligibility() {
    nextQOld = RealScalar.ZERO;
    /* eligibility trace vector is initialized to zero at the beginning of the
     * episode */
    z = Array.zeros(featureSize);
  }

  @Override
  public StateActionCounter sac() {
    return sac;
  }
}
