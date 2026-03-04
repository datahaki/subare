// code by jph
package ch.alpine.subare.alg;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.io.TableBuilder;

public abstract class BaseIteration {
  private final TableBuilder tableBuilder = new TableBuilder();
  private int iterations = 0;

  protected void appendRow(Scalar delta) {
    tableBuilder.appendRow(RealScalar.of(iterations), delta);
    ++iterations;
  }

  public final TableBuilder tableBuilder() {
    return tableBuilder;
  }

  public final int iterations() {
    return iterations;
  }
}
