// code by gemini, jpg
package ch.alpine.subare.net.los;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collector;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Transpose;

public enum NetLossCollectors {
  ;
  public record AccumulatedNetLoss(Scalar loss, List<Tensor> gradOutputs) {
    public Tensor compile() {
      return Tensor.of(gradOutputs.stream(), gradOutputs.size());
    }

    public Tensor compileBatch() {
      return Transpose.of(compile());
    }
  }

  public static Collector<NetLoss, ?, AccumulatedNetLoss> toAccumulated() {
    return Collector.of(Accumulator::new, Accumulator::accept, Accumulator::combine, Accumulator::finishAccumulated);
  }

  private static final class Accumulator {
    Scalar loss = RealScalar.ZERO;
    final List<Tensor> gradients = new ArrayList<>();

    void accept(NetLoss netLoss) {
      // Objects.requireNonNull(item, "item");
      loss = loss.add(netLoss.loss());
      gradients.add(netLoss.gradOutput());
    }

    Accumulator combine(Accumulator other) {
      loss = loss.add(other.loss);
      gradients.addAll(other.gradients);
      return this;
    }

    AccumulatedNetLoss finishAccumulated() {
      return new AccumulatedNetLoss(loss, gradients);
    }
  }
}
