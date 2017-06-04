// code by jph
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.function.Function;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.StepDigestType;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteQsas;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EpisodeKickoff;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.sca.Round;

/** Sarsa applied to gambler */
class SD_Gambler {
  static Function<Scalar, Scalar> ROUND = Round.toMultipleOf(DecimalScalar.of(.1));

  static void handle(StepDigestType type) throws Exception {
    System.out.println(type);
    Gambler gambler = Gambler.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    int EPISODES = 30;
    Tensor epsilon = Subdivide.of(.1, .01, EPISODES);
    PolicyInterface policyInterface = new EquiprobablePolicy(gambler);
    DiscreteQsa qsa = DiscreteQsa.build(gambler);
    System.out.println(qsa.size());
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/gambler_qsa_" + type + ".gif"), 100);
    for (int index = 0; index < EPISODES; ++index) {
      Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + epsilon.Get(index).map(ROUND) + " " + error.map(ROUND));
      StepDigest stepDigest = type.supply(gambler, qsa, RealScalar.of(.1), policyInterface);
      for (int count = 0; count < 3; ++count) {
        ExploringStartsBatch.apply(gambler, stepDigest, policyInterface);
        policyInterface = GreedyPolicy.bestEquiprobable(gambler, qsa);
        policyInterface = EGreedyPolicy.bestEquiprobable(gambler, qsa, epsilon.Get(index));
      }
      gsw.append(ImageFormat.of(GamblerHelper.joinAll(gambler, qsa, ref)));
    }
    gsw.close();
    qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    EpisodeInterface mce = EpisodeKickoff.single(gambler, policyInterface);
    while (mce.hasNext()) {
      StepInterface stepInterface = mce.step();
      Tensor state = stepInterface.prevState();
      System.out.println(state + " then " + stepInterface.action());
    }
  }

  public static void main(String[] args) throws Exception {
    handle(StepDigestType.original);
    handle(StepDigestType.expected);
    handle(StepDigestType.qlearning);
  }
}
