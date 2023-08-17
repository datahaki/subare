// code by jph
package ch.alpine.subare.book.ch06.maxbias;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ActionValueStatistics;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;

enum RSTQP_Maxbias {
  ;
  public static void main(String[] args) {
    Maxbias maxbias = new Maxbias(3);
    DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias);
    DiscreteQsa qsa = DiscreteQsa.build(maxbias);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        maxbias, qsa, DefaultLearningRate.of(3, 0.51));
    ActionValueStatistics avs = new ActionValueStatistics(maxbias);
    int batches = 5000;
    for (int index = 0; index < 500; ++index)
      TabularSteps.batch(maxbias, maxbias, rstqp, avs);
    Infoline.print(maxbias, batches, ref, qsa);
    System.out.println("---");
    ActionValueIteration avi = ActionValueIteration.of(maxbias, avs);
    avi.untilBelow(RealScalar.of(.0001));
    DiscreteUtils.print(avi.qsa());
    {
      Scalar error = DiscreteValueFunctions.distance(ref, avi.qsa());
      System.out.println("avs error=" + error);
    }
  }
}
