// code by jph
package ch.alpine.subare.book.ch01.tic;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/* package */ class Analysis {
  public static void showVariance() {
    Agent p1 = new Agent(1);
    p1.loadPolicy();
    Estimation estimation = p1.estimation;
    Set<State> set = AllStates.INSTANCE.getAll();
    @SuppressWarnings("unused")
    int count = 0;
    List<Double> list = new ArrayList<>();
    while (!set.isEmpty()) {
      Set<State> subset = new HashSet<>();
      {
        State state;
        state = set.iterator().next();
        for (int i = 0; i < 4; ++i) {
          state = state.getRotated();
          subset.add(state);
          set.remove(state);
        }
        state = state.getMirrored();
        for (int i = 0; i < 4; ++i) {
          state = state.getRotated();
          subset.add(state);
          set.remove(state);
        }
      }
      // if (1 == subset.size() ) {
      // System.out.println(subset.iterator().next());
      // }
      if (1 < subset.size()) {
        double[] data = //
            subset.stream().mapToDouble(estimation::get).toArray();
        // System.out.println("-----");
        // System.out.println(subset.iterator().next());
        for (@SuppressWarnings("unused")
        State state : subset) {
          // System.out.println(estimation.get(state));
        }
        Statistics statistics = new Statistics(data);
        double var = statistics.getVariance();
        list.add(var);
      }
      //
      ++count;
    }
    // average variance
    // 0.1 = 0.03338031055070961
    DoubleSummaryStatistics stats = list.stream().collect(Collectors.summarizingDouble(i -> i));
    System.out.println(stats.getAverage());
  }

  public static void main(String[] args) {
    Agent p1 = new Agent(1);
    p1.loadPolicy();
    Estimation estimation = p1.estimation;
    @SuppressWarnings("unused")
    Set<State> set = AllStates.INSTANCE.getAll();
    for (State state : AllStates.INSTANCE.getEquivalenceSet()) {
      System.out.println(state);
      System.out.println(estimation.get(state));
    }
    {
      State state = AllStates.INSTANCE.getFromHash(8338);
      System.out.println(state);
      System.out.println(estimation.get(state));
    }
  }
}
