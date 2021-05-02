// code by jph
package ch.alpine.subare.demo.prison;

import java.io.IOException;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.subare.ch02.Agent;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Put;

/* package */ enum AllPairsExport {
  ;
  public static void main(String[] args) throws IOException {
    // List<Supplier<Agent>> list = AgentSupplier.getOptimists(.01, .8, 30);
    // List<Supplier<Agent>> list = AgentSupplier.getUCBs(0, 6, 30);
    List<Supplier<Agent>> list = AgentSupplier.getEgreedyC(0.1, .8, 20);
    Put.of(HomeDirectory.file("egreedyc"), AllPairs.performance(list, 20, 500));
    System.out.println("done.");
  }
}
