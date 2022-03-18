// code by jph
package ch.alpine.subare.demo.streets;

import java.util.Arrays;
import java.util.List;

import ch.alpine.subare.ch02.Agent;
import ch.alpine.subare.ch02.FairArgAgent;
import ch.alpine.subare.ch02.OptimistAgent;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

class Judger {
  final Network network;
  final List<Agent> list;

  @SafeVarargs
  Judger(Network network, Agent... agents) {
    this.network = network;
    list = Arrays.asList(agents);
  }

  void play() {
    network.reset();
    // agents choose path
    for (Agent agent : list)
      network.feedAction(agent.takeAction());
    Tensor cost = network.cost();
    // network computes costs
    for (Agent agent : list) {
      int k = agent.getActionReminder();
      Scalar total = RealScalar.ZERO;
      for (int s : network.streetsFromAction(k))
        total = total.add(cost.Get(s));
      agent.feedback(k, total);
    }
  }

  public static void main(String[] args) {
    FairArgAgent a1 = new OptimistAgent(3, RealScalar.of(5), RealScalar.of(.1));
    FairArgAgent a2 = new OptimistAgent(3, RealScalar.of(5), RealScalar.of(.1));
    FairArgAgent a3 = new OptimistAgent(3, RealScalar.of(5), RealScalar.of(.1));
    a1.setOpeningSequence(0);
    a2.setOpeningSequence(1);
    a3.setOpeningSequence(2);
    Agent[] agents = new Agent[] { a1, a2, a3 };
    Judger judger = new Judger(new BridgeNetwork(), agents);
    for (int rnd = 0; rnd < 1000; ++rnd)
      judger.play();
    for (Agent a : agents)
      System.out.println(a.getRewardAverage());
    // System.out.println(a2.getRewardTotal());
    // System.out.println(a3.getRewardTotal());
  }
}
