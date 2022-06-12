// code by jph
package ch.alpine.subare.demo.prison;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.subare.ch02.Agent;
import ch.alpine.subare.ch02.EGreedyAgent;
import ch.alpine.subare.ch02.GradientAgent;
import ch.alpine.subare.ch02.OptimistAgent;
import ch.alpine.subare.ch02.RandomAgent;
import ch.alpine.subare.ch02.UCBAgent;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;

/* package */ enum AgentSupplier {
  ;
  public static final List<Supplier<Agent>> mixed = List.of( //
      // () -> new ConstantAgent(2, 0), //
      // () -> new ConstantAgent(2, 1), //
      () -> new TitForTatAgent(), //
      () -> new RandomAgent(2), //
      () -> new EGreedyAgent(2, i -> RationalScalar.of(1, 5), "1/5"), //
      () -> new EGreedyAgent(2, i -> RationalScalar.of(1, i.number().intValue() + 1), "1/i"), //
      () -> new GradientAgent(2, RealScalar.of(.1)), //
      () -> new OptimistAgent(2, RealScalar.of(6), RealScalar.of(.1)), //
      () -> new UCBAgent(2, RealScalar.of(1))) //
  ;

  public static List<Supplier<Agent>> getUCBs(double cLo, double cHi, int steps) {
    List<Supplier<Agent>> list = new ArrayList<>();
    for (double c = cLo; c <= cHi; c += (cHi - cLo) / (steps - 1)) {
      Scalar cs = RealScalar.of(c);
      list.add(() -> new UCBAgent(2, cs));
    }
    return list;
  }

  public static List<Supplier<Agent>> getOptimists(double cLo, double cHi, int steps) {
    List<Supplier<Agent>> list = new ArrayList<>();
    for (double c = cLo; c <= cHi; c += (cHi - cLo) / (steps - 1)) {
      Scalar cs = RealScalar.of(c);
      list.add(() -> new OptimistAgent(2, RealScalar.of(6), cs));
    }
    return list;
  }

  public static List<Supplier<Agent>> getEgreedyC(double cLo, double cHi, int steps) {
    List<Supplier<Agent>> list = new ArrayList<>();
    for (double c = cLo; c <= cHi; c += (cHi - cLo) / (steps - 1)) {
      Scalar cs = RealScalar.of(c);
      Supplier<Agent> sup = () -> new EGreedyAgent(2, i -> cs, cs.toString());
      list.add(sup);
    }
    return list;
  }
}
