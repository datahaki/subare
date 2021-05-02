// code by fluric
package ch.alpine.subare.analysis;

import java.util.function.Supplier;

import ch.alpine.subare.ch04.gambler.GamblerModel;
import ch.alpine.subare.ch04.grid.Gridworld;
import ch.alpine.subare.ch05.infvar.InfiniteVariance;
import ch.alpine.subare.ch05.racetrack.RacetrackHelper;
import ch.alpine.subare.ch05.wireloop.WireloopHelper;
import ch.alpine.subare.ch05.wireloop.WireloopReward;
import ch.alpine.subare.ch06.cliff.Cliffwalk;
import ch.alpine.subare.ch06.maxbias.Maxbias;
import ch.alpine.subare.ch06.windy.Windygrid;
import ch.alpine.subare.ch08.maze.DynamazeHelper;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.demo.airport.Airport;
import ch.alpine.subare.demo.virtualstations.VirtualStations;
import ch.alpine.tensor.RationalScalar;

public enum MonteCarloExamples implements Supplier<MonteCarloInterface> {
  AIRPORT(() -> Airport.INSTANCE), //
  VIRTUALSTATIONS(() -> VirtualStations.INSTANCE), //
  GAMBLER_20(() -> new GamblerModel(20, RationalScalar.of(4, 10))), //
  GAMBLER_100(() -> GamblerModel.createDefault()), //
  MAZE2(() -> DynamazeHelper.original("maze2")), //
  MAZE5(() -> DynamazeHelper.create5(3)), //
  WIRELOOP_4(() -> {
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    return WireloopHelper.create("wire4", WireloopReward::id_x, wireloopReward);
  }), //
  WIRELOOP_5(() -> {
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    return WireloopHelper.create("wire5", WireloopReward::id_x, wireloopReward);
  }), //
  WIRELOOP_C(() -> {
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    return WireloopHelper.create("wirec", WireloopReward::id_x, wireloopReward);
  }), //
  GRIDWORLD(() -> new Gridworld()), //
  INFINITEVARIANCE(() -> new InfiniteVariance()), //
  RACETRACK(() -> RacetrackHelper.create("track0", 5)), //
  CLIFFWALK(() -> new Cliffwalk(12, 4)), //
  MAXBIAS(() -> new Maxbias(5)), //
  WINDYGRID(() -> Windygrid.createFour()), //
  ;

  private final Supplier<MonteCarloInterface> supplier;

  private MonteCarloExamples(Supplier<MonteCarloInterface> supplier) {
    this.supplier = supplier;
  }

  @Override
  public MonteCarloInterface get() {
    return supplier.get();
  }
}
