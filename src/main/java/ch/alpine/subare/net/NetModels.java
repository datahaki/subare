// code by jph
package ch.alpine.subare.net;

import java.util.List;

import ch.alpine.subare.net.lay.ArgMaxLayer;
import ch.alpine.subare.net.lay.SoftmaxLayer;

public enum NetModels {
  ;
  /** the output of netLayer is a vector of length n
   * where n corresponds to the number of labels for classification
   * 
   * @param netLayer
   * @return layer that outputs vector of length 1 with integer value as entry */
  public static NetModel softmax_argMax(NetLayer netLayer) {
    ArgMaxLayer argMaxLayer = new ArgMaxLayer();
    return new NetModel( //
        NetChain.join( //
            netLayer, //
            SoftmaxLayer.INSTANCE, //
            argMaxLayer), //
        argMaxLayer.crossEntropyLoss());
  }
  // public static NetModel argMax(NetLayer netLayer) {
  // ArgMaxLayer argMaxLayer = new ArgMaxLayer();
  // return new NetModel( //
  // NetChain.join( //
  // netLayer, //
  // argMaxLayer), //
  // argMaxLayer.crossEntropyLoss());
  // }

  /** @param netLayer
   * @return */
  public static List<String> toString(NetModel netModel) {
    List<String> list = NetLayers.toString(netModel.netLayer());
    list.add(netModel.lossFunction().toString());
    return list;
  }
}
