// code by jph
package ch.alpine.subare.pol;

import ch.alpine.subare.math.FairArg;
import ch.alpine.subare.math.Index;
import ch.alpine.subare.mod.DiscreteModel;
import ch.alpine.subare.mod.StandardModel;
import ch.alpine.subare.util.UcbUtils;
import ch.alpine.subare.val.QsaInterface;
import ch.alpine.subare.val.VsInterface;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;

/** upper confidence bound is greedy except that it encourages
 * exploration if an action has not been encountered often relative to other actions */
class UcbPolicy extends PolicyBase {
  UcbPolicy(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    super(discreteModel, qsa, sac);
  }

  UcbPolicy(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    super(standardModel, vs, sac);
  }

  @Override // from PolicyBase
  public Tensor getBestActions(Tensor state) {
    Tensor actions = discreteModel.actions(state);
    Tensor va = Tensor.of(actions.stream().parallel() //
        .map(action -> UcbUtils.getUpperConfidenceBound(state, action, qsa.value(state, action), sac)));
    FairArg fairArg = FairArg.max(va);
    return Tensor.of(fairArg.options().stream().map(actions::get));
  }

  @Override // from Policy
  public Distribution getDistribution(Tensor state) {
    Tensor bestActions = getBestActions(state);
    Index index = Index.build(bestActions);
    final int optimalCount = bestActions.length();
    Tensor pdf = Tensor.of(discreteModel.actions(state).stream() //
        .map(action -> index.containsKey(action) ? Rational.of(1, optimalCount) : RealScalar.ZERO));
    return CategoricalDistribution.fromUnscaledPDF(pdf);
  }

  @Override // from Policy
  public Scalar probability(Tensor state, Tensor action) {
    Tensor actions = getBestActions(state);
    return actions.stream().anyMatch(action::equals) // computational complexity is O(n)
        ? Rational.of(1, actions.length())
        : RealScalar.ZERO;
  }

  @Override // from PolicyBase
  public UcbPolicy copy() {
    return new UcbPolicy(discreteModel, qsa, sac);
  }
}
