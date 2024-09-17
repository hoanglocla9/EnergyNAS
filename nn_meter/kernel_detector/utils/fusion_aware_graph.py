# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import networkx as nx
from .union_find import UF
from nn_meter.utils.graph_tool import ModelGraph


class FusionAwareGraph:
    def __init__(self, model_graph: ModelGraph):
        self._model_graph = model_graph
        self._dag = list(nx.topological_sort(model_graph.get_networkx_graph()))
        self._uf = UF(len(self._dag))

        reverse = {}
        for index, name in enumerate(self._dag):
            reverse[name] = index
        outbounds = []
        inbounds = []
        for index, name in enumerate(self._dag):
            outbounds.append(
                {reverse[outbound]
                    for outbound in self._model_graph.get_node_outbounds(name)}
            )
            inbounds.append(
                {reverse[inbound]
                    for inbound in self._model_graph.get_node_inbounds(name)}
            )

        self._outbounds = outbounds
        self._inbounds = inbounds
        self._ready = [not inbounds[i] for i in range(0, len(self))]
        self._types = [model_graph.get_node_type(name) for name in self._dag]
        self.bbs = []

    @property
    def nodes(self):
        return self._dag

    def __len__(self):
        return len(self._dag)

    def __getitem__(self, key):
        return self._dag[key]

    def fuse(self, node, outnode, update=False):
        """
        node should be root, outnode should be an unfused single node
        """
        self._uf.union(node, outnode)
        if not update:
            self._outbounds[node] = self._outbounds[outnode]
        else:
            self._outbounds[node].update(self._outbounds[outnode])

    def mark_ready(self, node):
        self._ready[node] = True

    def is_ready(self, node):
        for inbound in self._inbounds[node]:
            if not self.is_ready[inbound]:
                return False
        return True

    def is_visited(self, node):
        return self._ready[node]

    def get_outbounds(self, node):
        return self._outbounds[node]

    def get_inbounds(self, node):
        return self._inbounds[node]

    def get_type(self, node):
        return self._types[node]

    def get_basicblocks(self):
        bbs = []

        for _ in range(0, len(self)):
            bbs.append([])

        for i in range(0, len(self)):
            root = self._uf.find(i)
            bbs[root].append(self[i])

        bbs = [bb for bb in bbs if bb]
        self.bbs = bbs
        return bbs

    def find_root(self, node):
        return self[self._uf.find(node)]

    def is_fused(self, node):
        return self._uf.find(node) != node

    def is_connected(self, p, q):
        return self._uf.connected(p, q)

    def get_type_of_node(self, node):
        node_name = self[node]
        return self._model_graph.graph[node_name]['attr']['type'] if 'type' in self._model_graph.graph[node_name]['attr'] else None

    def map_conv_node_to_parameter_based_name(self, conv_op: int):
        layer_name = self[conv_op]
        parameter = self._model_graph.graph[layer_name]['attr']['attr']
        ks = parameter['ks']
        dilation = parameter['dilations']
        group = parameter['group']
        strides = parameter['strides']
        pads = parameter['pads']
        return f"{ks}_{dilation}_{group}_{strides}_{pads}"

    def get_fusion_type(self, block):
        fusion_type = '-'.join(self.get_type_of_node(node) for node in block)
        return fusion_type

    # input should be conv_node
    def get_fusion_block_of_node(self, conv_node, bbs):
        return self.get_fusion_type(bbs[conv_node])

    def get_parallelable_blocks(self):
        bbs = []
        for _ in range(0, len(self)):
            bbs.append([])

        for i in range(0, len(self)):
            root = self._uf.find(i)
            bbs[root].append(i)

        previous_node_to_conv_mapping = {"conv": {}, "dwconv": {}}
        for block in bbs:
            if len(block) == 0:
                continue

            root = self._uf.find(block[0])
            inbound_of_block = self.get_inbounds(root)
            outbound_of_block = self.get_outbounds(root)

            for op in block:
                outbound_of_op = self.get_outbounds(op)
                outbound_of_block.union(outbound_of_op)

            for op in block:
                op_type = self.get_type_of_node(op)
                if op_type in ["conv", "dwconv"]:
                    if len(inbound_of_block) == 1:
                        inbound_node = list(inbound_of_block)[0]
                        if not inbound_node in previous_node_to_conv_mapping[op_type]:
                            previous_node_to_conv_mapping[op_type][inbound_node] = [
                                op]
                        else:
                            previous_node_to_conv_mapping[op_type][inbound_node].append(
                                op)

        parallelable_dict = {}
        for op_type in previous_node_to_conv_mapping:
            for previous_node in previous_node_to_conv_mapping[op_type]:
                if len(previous_node_to_conv_mapping[op_type][previous_node]) == 1:
                    continue

                parallelable_convs_set = set(
                    previous_node_to_conv_mapping[op_type][previous_node])

                list_of_conv_ops = previous_node_to_conv_mapping[op_type][previous_node]

                outbound_node_to_conv_ops = {}
                for conv in list_of_conv_ops:
                    outbound_nodes = self.get_outbounds(conv)
                    for outbound_node in outbound_nodes:
                        if outbound_node not in outbound_node_to_conv_ops:
                            outbound_node_to_conv_ops[outbound_node] = []
                        else:
                            outbound_node_to_conv_ops[outbound_node].append(
                                conv)

                for outbound_node in outbound_node_to_conv_ops:
                    inbound_nodes_of_this_outbound_node = self.get_inbounds(
                        outbound_node)
                    if inbound_nodes_of_this_outbound_node != set(outbound_node_to_conv_ops[outbound_node]):
                        for removed_conv_ops in outbound_node_to_conv_ops[outbound_node]:
                            parallelable_convs_set.remove(removed_conv_ops)

                # Check parameters of conv nodes
                grouped_conv_set_by_parameters = {}
                for conv in parallelable_convs_set:
                    parameter_based_name = self.map_conv_node_to_parameter_based_name(
                        conv)
                    fusion_type = self.get_fusion_block_of_node(conv, bbs)
                    if parameter_based_name not in grouped_conv_set_by_parameters:

                        grouped_conv_set_by_parameters[parameter_based_name] = {fusion_type: [
                            conv]}
                    else:
                        if fusion_type not in grouped_conv_set_by_parameters[parameter_based_name]:
                            grouped_conv_set_by_parameters[parameter_based_name][fusion_type] = [
                                conv]
                        else:
                            grouped_conv_set_by_parameters[parameter_based_name][fusion_type].append(
                                conv)

                for group in grouped_conv_set_by_parameters:
                    for fusion_type in grouped_conv_set_by_parameters[group]:
                        if len(grouped_conv_set_by_parameters[group][fusion_type]) > 1:
                            tmp = []
                            for conv in grouped_conv_set_by_parameters[group][fusion_type]:
                                block_of_conv = bbs[conv]
                                tmp.append([self[op] for op in block_of_conv])
                            if fusion_type not in parallelable_dict:
                                parallelable_dict[fusion_type] = [tmp]
                            else:
                                parallelable_dict[fusion_type].append(tmp)
        return parallelable_dict
