package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.Objects;

import ch.alpine.tensor.ext.Integers;

public record LayerSlice(int d_in, int d_out, NetLayer netLayer) implements Serializable {
  public LayerSlice {
    Integers.requirePositiveOrZero(d_in);
    Integers.requirePositiveOrZero(d_out);
    Objects.requireNonNull(netLayer);
  }
}
