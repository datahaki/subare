// code by fluric
package ch.alpine.subare.td;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.PolicyExt;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.math.Coinflip;
import ch.alpine.subare.util.FeatureQsaAdapter;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.StateAction;
import ch.alpine.subare.util.StateActionCounterUtil;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.red.Mean;

public class DoubleTrueOnlineSarsa extends AbstractTrueOnlineSarsa {
  private final Coinflip coinflip = Coinflip.fair();
  // ---
  private final StateActionCounter sac1;
  private final StateActionCounter sac2;
  private final PolicyExt policy1;
  private final PolicyExt policy2;
  // ---
  /** feature weight vectors w1 and w2 are a long-term memory, accumulating over the lifetime of the system */
  private final FeatureWeight w1;
  private final FeatureWeight w2;
  // ---
  private Scalar nextQOld;
  /** eligibility trace z is a short-term memory, typically lasting less time than the length of an episode */
  private Tensor z;

  public DoubleTrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, SarsaEvaluation evaluationType, Scalar lambda, //
      FeatureMapper featureMapper, //
      LearningRate learningRate, //
      FeatureWeight w1, FeatureWeight w2, //
      StateActionCounter sac1, StateActionCounter sac2, //
      PolicyExt policy1, PolicyExt policy2) {
    super(monteCarloInterface, evaluationType, lambda, learningRate, featureMapper);
    this.sac1 = sac1;
    this.sac2 = sac2;
    this.policy1 = policy1;
    this.policy2 = policy2;
    this.w1 = w1;
    this.w2 = w2;
    resetEligibility();
  }

  /** faster when only part of the qsa is required */
  @Override
  public final QsaInterface qsaInterface() {
    return new FeatureQsaAdapter(getW(), featureMapper);
  }

  /** faster when only part of the qsa is required */
  private QsaInterface qsaInterface(Tensor w) {
    return new FeatureQsaAdapter(w, featureMapper);
  }

  /** @return policy with respect to (w1 + w2) / 2 and sac1+sac2 */
  public PolicyBase getPolicy() {
    PolicyBase copy = (PolicyBase) policy1.copy();
    copy.setQsa(qsaInterface());
    copy.setSac(StateActionCounterUtil.getSummedSac(sac1, sac2, monteCarloInterface));
    return copy;
  }

  /** @return unmodifiable weight vector w */
  @Override
  public final Tensor getW() {
    return Mean.of(Tensors.of(w1.get(), w2.get())).unmodifiable();
  }

  @Override // from StepDigest
  public final void digest(StepRecord stepRecord) {
    ((PolicyBase) policy1).setQsa(qsaInterface(w1.get()));
    ((PolicyBase) policy2).setQsa(qsaInterface(w2.get()));
    // randomly select which w to read and write
    boolean flip = coinflip.tossHead(); // flip coin, probability 0.5 each
    FeatureWeight W1 = flip ? w2 : w1;
    StateActionCounter Sac1 = flip ? sac2 : sac1; // for updating
    PolicyExt Policy1 = flip ? policy1 : policy2;
    PolicyExt Policy2 = flip ? policy2 : policy1;
    // ---
    Tensor prevState = stepRecord.prevState();
    Tensor prevAction = stepRecord.action();
    Tensor nextState = stepRecord.nextState();
    // ---
    Scalar reward = monteCarloInterface.reward(prevState, prevAction, nextState);
    // ---
    Scalar alpha = learningRate.alpha(stepRecord, Sac1);
    Scalar alpha_gamma_lambda = alpha.multiply(gamma_lambda);
    Tensor x = featureMapper.getFeature(StateAction.key(prevState, prevAction));
    Scalar prevQ = (Scalar) W1.get().dot(x);
    Scalar nextQ = evaluationType.crossEvaluate(nextState, Policy1, Policy2);
    Scalar delta = reward.add(gamma.multiply(nextQ)).subtract(prevQ);
    // eq (12.11)
    z = z.multiply(gamma_lambda) //
        .add(x.multiply(RealScalar.ONE.subtract(z.dot(x).multiply(alpha_gamma_lambda))));
    // ---
    Scalar diffQ = prevQ.subtract(nextQOld);
    Tensor scalez = z.multiply(alpha.multiply(delta.add(diffQ)));
    Tensor scalex = x.multiply(alpha.multiply(diffQ));
    if (flip)
      w2.set(w2.get().add(scalez).subtract(scalex));
    else
      w1.set(w1.get().add(scalez).subtract(scalex));
    nextQOld = nextQ;
    // ---
    Sac1.digest(stepRecord);
    // ---
    if (monteCarloInterface.isTerminal(nextState))
      resetEligibility();
  }

  private void resetEligibility() {
    nextQOld = RealScalar.ZERO;
    /* eligibility trace vector is initialized to zero at the beginning of the episode */
    z = Array.zeros(featureMapper.featureSize());
  }

  @Override
  public StateActionCounter sac() {
    return StateActionCounterUtil.getSummedSac(sac1, sac2, monteCarloInterface);
  }
}
