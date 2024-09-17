# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.utils.graph_tool import ModelGraph
from .utils.constants import DUMMY_TYPES
from .utils.ir_tools import convert_nodes, remove_cast_layers
from .rule_reader import RuleReader
from .rule_splitter import RuleSplitter


class KernelDetector:
    def __init__(self, rule_file: str, useParallel: bool = False):
        self.reader = RuleReader(rule_file)
        self.splitter = RuleSplitter(self.reader)
        self.model_graph = None
        self.bbs = []
        self._global_index = 0
        self.useParallel = useParallel

    def load_graph(self, graph):
        # new_graph = remove_cast_layers(graph)
        new_graph = convert_nodes(graph)
        self.model_graph = ModelGraph(graph=new_graph)
        self.model_graph.refresh()
        self.bbs = self.splitter.split(self.model_graph)
        if self.useParallel:
            self.parallelable_blocks_dict = self.splitter._fusion_graph.get_parallelable_blocks()
            self.reformat_basicblocks_and_parallelable_blocks()
        else:
            tmp = self.bbs
            self.bbs = {}
            self.bbs['basic'] = tmp

    def reformat_basicblocks_and_parallelable_blocks(self):
        results = {"parallelable": [blocks for blocks_list in self.parallelable_blocks_dict.values()
                                    for blocks in blocks_list],
                   "basic": []}
        # faltten the parallelable_blocks_dict

        parallelable_blocks_list = [b for blocks_list in self.parallelable_blocks_dict.values()
                                    for blocks in blocks_list for b in blocks]

        for bb in self.bbs:
            if bb not in parallelable_blocks_list:
                results["basic"].append(bb)
        self.bbs = results

    def get_kernels(self):
        kernels = []
        self._global_index = 0
        self._layer_kernel_dict = {}
        # print(self.bbs['basic'])

        for bb in self.bbs['basic']:
            kernel = self._bb_to_kernel_basic(bb)
            if kernel is not None:
                kernels.append(kernel)
            self._global_index += 1

        if self.useParallel:
            for parallel_blocks in self.bbs['parallelable']:
                kernel = self._bb_to_kernel_parallel(parallel_blocks)
                if kernel is not None:
                    kernels.append(kernel)
                self._global_index += 1

        # print("before: ", kernels)
        # self._fetch_connections(kernels)
        # print("===========================")
        # print("after:", kernels)
        return kernels

    def _fetch_connections(self, kernels):
        fusion_graph = self.splitter._fusion_graph

        for kernel in kernels:
            kernel["inbounds"] = []

        for i in range(len(fusion_graph)):
            layer = fusion_graph[i]
            kernel = self._layer_kernel_dict.get(layer)

            if kernel:
                outbounds = [fusion_graph.find_root(
                    outbound) for outbound in fusion_graph.get_outbounds(i)]
                outbounds = [self._layer_kernel_dict[outbound]
                             for outbound in outbounds]

                for outbound in outbounds:
                    outbound["inbounds"].append(kernel["name"])

                outbounds = [outbound["name"] for outbound in outbounds]
                kernel["outbounds"] = outbounds

    def _bb_to_kernel_parallel(self, parallel_blocks):
        types = [self.model_graph.get_node_type(
            node) for node in parallel_blocks[0]]  # types is similar for other blocks in the same parallel blocks.
        types = [t for t in types if t and t not in DUMMY_TYPES]

        if types:
            type = "-".join(types)
            name = f"{type}#{self._global_index}"
            kernel = {
                "op": type,
                "name": name
            }
            input_shapes, output_shapes = [], []
            for bb in parallel_blocks:
                layer = bb[0]
                self._layer_kernel_dict[layer] = kernel
                attr = self.model_graph.get_node_attr(layer)["attr"]
                input_shapes.append(self.model_graph.get_node_attr(layer)[
                    "input_shape"])
                output_shapes.append(self.model_graph.get_node_attr(layer)[
                    "output_shape"])
            # input shape is similar for all parallel blocks:
            input_shape = input_shapes[0]
            # should only have a single shape for input
            # assert len(list(set(input_shapes))) > 1
            # output shape is a concatnate of all parallel blocks with axis of out channel
            output_shape = [[output_shapes[0][0][0], output_shapes[0][0][1],
                            output_shapes[0][0][2], sum(s[0][3] for s in output_shapes)]]
            # other attr should be similar
            attr = self.model_graph.get_node_attr(
                parallel_blocks[0][0])["attr"]
            kernel["input_tensors"] = input_shape

            if "ks" in attr:
                kernel["ks"] = attr["ks"]
            if "strides" in attr:
                kernel["strides"] = attr["strides"]
            if "split_dim" in attr:
                kernel["split_dim"] = attr["split_dim"]
            if "pads" in attr:
                kernel["pads"] = attr["pads"]

            if len(input_shape) >= 1:
                if len(input_shape[0]) == 4:
                    kernel["inputh"] = input_shape[0][1]
                    kernel["inputw"] = input_shape[0][2]
                kernel["cin"] = input_shape[0][-1]

            if len(output_shape) == 1:
                kernel["cout"] = output_shape[0][-1]
            elif len(output_shape) > 1:
                kernel["output_tensors"] = output_shape
            print(kernel)
            return kernel
        else:
            return None

    def _bb_to_kernel_basic(self, bb):
        types = [self.model_graph.get_node_type(node) for node in bb]
        # logging.info(types)
        types = [t for t in types if t and t not in DUMMY_TYPES]

        if types:
            type = "-".join(types)
            name = f"{type}#{self._global_index}"

            kernel = {
                "op": type,
                "name": name,
            }

            layer = bb[0]
            self._layer_kernel_dict[layer] = kernel
            type = types[0]
            attr = self.model_graph.get_node_attr(layer)["attr"]
            input_shape = self.model_graph.get_node_attr(layer)["input_shape"]
            output_shape = self.model_graph.get_node_attr(layer)[
                "output_shape"]

            # Remove const from first biasadd of hswish
            if type == "hswish":
                input_shape = [input_shape[0]]
            kernel["input_tensors"] = input_shape

            if "ks" in attr:
                kernel["ks"] = attr["ks"]
            if "strides" in attr:
                kernel["strides"] = attr["strides"]
            if "split_dim" in attr:
                kernel["split_dim"] = attr["split_dim"]
            if "pads" in attr:
                kernel["pads"] = attr["pads"]

            if len(input_shape) >= 1:
                if len(input_shape[0]) == 4:
                    kernel["inputh"] = input_shape[0][1]
                    kernel["inputw"] = input_shape[0][2]
                kernel["cin"] = input_shape[0][-1]

            if len(output_shape) == 1:
                kernel["cout"] = output_shape[0][-1]
            elif len(output_shape) > 1:
                kernel["output_tensors"] = output_shape

            return kernel
        else:
            return None
