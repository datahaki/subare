// code by jph
package ch.alpine.subare.net.mod;

import java.io.Serializable;
import java.util.Objects;

import ch.alpine.tensor.ext.Integers;

public record ModelSlice(int d_in, int d_out, NetModel netModel) implements Serializable {
  public ModelSlice {
    Integers.requirePositiveOrZero(d_in);
    Integers.requirePositiveOrZero(d_out);
    Objects.requireNonNull(netModel);
  }
}
