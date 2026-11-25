// code by jph
package ch.alpine.subare.net.los;

import java.io.Serializable;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public record NetLoss(Scalar loss, Tensor gradOutput) implements Serializable {
}
