// code by fluric
package ch.alpine.subare.demo;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowDialog;
import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.analysis.DiscreteModelErrorAnalysis;
import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.demo.airport.Airport;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.td.TrueOnlineSarsa;
import ch.alpine.subare.util.ConstantExplorationRate;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Timing;

/** uses TrueOnlineSarsa */
/* package */ enum AirportShow {
  ;
  public static void main(String[] args) throws Exception {
    Tensor XYmc = Tensors.empty();
    Tensor XYsarsa = Tensors.empty();
    Tensor XYtoSarsa = Tensors.empty();
    Airport airport = Airport.INSTANCE;
    DiscreteQsa optimalQsa = ActionValueIterations.solve(airport, RealScalar.of(0.0001));
    // DiscreteUtils.print(optimalQsa);
    // Policies.print(policyQsa, airport.states());
    final int batches = 10;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(airport);
    {
      Timing timing = Timing.started();
      for (int index = 0; index < batches; ++index) {
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(airport, mces.qsa(), mces.sac());
        policy.setExplorationRate(ConstantExplorationRate.of(0.1));
        ExploringStarts.batch(airport, policy, mces);
        XYmc.append(Tensors.of(RealScalar.of(index), DiscreteModelErrorAnalysis.LINEAR_POLICY.getError(airport, optimalQsa, mces.qsa())));
      }
      System.out.println("time for MonteCarlo: " + timing.seconds() + "s");
      // Policies.print(GreedyPolicy.bestEquiprobable(airport, mces.qsa()), airport.states());
    }
    DiscreteQsa qsaSarsa = DiscreteQsa.build(airport); // q-function for training, initialized to 0
    SarsaType sarsaType = SarsaType.ORIGINAL;
    StateActionCounter sac = new DiscreteStateActionCounter();
    final Sarsa sarsa = sarsaType.sarsa(airport, ConstantLearningRate.of(RealScalar.of(0.05)), qsaSarsa, sac,
        PolicyType.EGREEDY.bestEquiprobable(airport, qsaSarsa, sac));
    {
      Timing timing = Timing.started();
      for (int index = 0; index < batches; ++index) {
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(airport, sarsa.qsa(), sarsa.sac());
        policy.setExplorationRate(ConstantExplorationRate.of(0.1));
        ExploringStarts.batch(airport, policy, 1, sarsa);
        XYsarsa.append(Tensors.of(RealScalar.of(index), DiscreteModelErrorAnalysis.LINEAR_POLICY.getError(airport, optimalQsa, sarsa.qsa())));
      }
      System.out.println("time for Sarsa: " + timing.seconds() + "s");
    }
    // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
    LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.2));
    FeatureMapper mapper = ExactFeatureMapper.of(airport);
    FeatureWeight w = new FeatureWeight(mapper);
    StateActionCounter toSac = new DiscreteStateActionCounter();
    TrueOnlineSarsa toSarsa = SarsaType.ORIGINAL.trueOnline(airport, RealScalar.of(0.7), mapper, learningRate, w, toSac,
        PolicyType.EGREEDY.bestEquiprobable(airport, DiscreteQsa.build(airport), toSac));
    {
      Timing timing = Timing.started();
      for (int index = 0; index < batches; ++index) {
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(airport, toSarsa.qsa(), toSarsa.sac());
        policy.setExplorationRate(ConstantExplorationRate.of(0.1));
        ExploringStarts.batch(airport, policy, toSarsa);
        DiscreteQsa toQsa = toSarsa.qsa();
        XYtoSarsa.append(Tensors.of(RealScalar.of(index), DiscreteModelErrorAnalysis.LINEAR_POLICY.getError(airport, optimalQsa, toQsa)));
      }
      System.out.println("time for TrueOnlineSarsa: " + timing.seconds() + "s");
    }
    {
      Show show = new Show();
      show.add(ListLinePlot.of(XYmc)).setLabel("MonteCarlo");
      show.add(ListLinePlot.of(XYsarsa)).setLabel("Sarsa");
      show.add(ListLinePlot.of(XYtoSarsa)).setLabel("TrueOnlineSarsa");
      ShowDialog.of(show);
    }
    // DiscreteQsa toQsa = toSarsa.qsa();
    // System.out.println(toSarsa.getW());
    // toSarsa.printValues();
    // toSarsa.printPolicy();
  }
}
