// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.function.Function;

import ch.alpine.subare.net.lay.NetLayer;

public interface NetOptimizerFactory extends Function<NetLayer, NetOptimizer>, Serializable {
}
