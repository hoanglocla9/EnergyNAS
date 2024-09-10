from .model import *
from nni.retiarii import Mutator


def update_node_attribute(node, input_shape, output_shape):
    if node.operation.attributes.get("input_shape") is None:
        node.operation.attributes["input_shape"] = input_shape
    if node.operation.attributes.get("output_shape") is None:
        node.operation.attributes["output_shape"] = output_shape
    
class BlockMutator(Mutator):
    def __init__(self, target):
        super(BlockMutator, self).__init__()
        self.target = target

    def mutate(self, model):
        nodes = model.get_nodes_by_label(self.target)
        assert len(nodes) == 1
        node = nodes[0]
        node.name = self.target + "_0"
        graph = node.graph
        

        related_info = node.operation.parameters
        ## head parameters
        conv_kernel_size = self.choice(related_info['conv_kernel_size_options'])
        conv_n_filter = self.choice(related_info['conv_n_filter_options'])
        conv_activation_fn = self.choice(related_info['conv_activation_fn_options'])
        pooling_op_type = self.choice(related_info['pooling_op_type_options'])
        pooling_kernel_size = self.choice(related_info['pooling_kernel_size_options'])
        conv_n_layer = self.choice(related_info['conv_n_layer_options'])
        conv_op_type = related_info['conv_op_type']
        in_ch = related_info['in_ch']
        isJustFaltten = self.choice(related_info['flatten_options']) 
        # tail parameters
        n_neurons = self.choice(related_info['n_neurons_options'])
        fc_activation_fn = self.choice(related_info['fc_activation_fn_options'])
        fc_n_layer = self.choice(related_info['fc_n_layer_options'])
        fc_op_type = related_info['fc_op_type']

        if isJustFaltten == False:
            if conv_n_layer > 1:
                node.update_operation(conv_op_type, {
                    'kernel_size': conv_kernel_size,
                    'in_channels': 1,
                    'out_channels': conv_n_filter,
                    'padding': conv_kernel_size//2
                })
                update_node_attribute(node, [[1, 1, in_ch]], [[1, conv_n_filter, in_ch]])
                if "None" not in conv_activation_fn:
                    node = graph.insert_node_on_edge(node.outgoing_edges[0],
                            '{}_conv_activation_0'.format(self.target),
                            conv_activation_fn, {})
                update_node_attribute(node, [[1, conv_n_filter, in_ch]], [[1, conv_n_filter, in_ch]])
                for i in range(1, conv_n_layer):
                    if i != conv_n_layer-1:
                        node = graph.insert_node_on_edge(node.outgoing_edges[0],
                            '{}_conv_{}'.format(self.target, i),
                            conv_op_type, {
                                'kernel_size': conv_kernel_size,
                                'in_channels': conv_n_filter,
                                'out_channels': conv_n_filter,
                                'padding': conv_kernel_size//2
                            })
                        update_node_attribute(node, [[1, conv_n_filter, in_ch]], [[1, conv_n_filter, in_ch]])
                        if "None" not in conv_activation_fn:
                            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                                '{}_conv_activation_{}'.format(self.target, i),
                                conv_activation_fn, {})
                        update_node_attribute(node, [[1, conv_n_filter, in_ch]], [[1, conv_n_filter, in_ch]])
                    else:
                        node = graph.insert_node_on_edge(node.outgoing_edges[0],
                            '{}_{}'.format(self.target, i),
                            conv_op_type, {
                            'kernel_size': conv_kernel_size,
                            'in_channels': conv_n_filter,
                            'out_channels': 1,
                            'padding': conv_kernel_size//2
                        })
                        update_node_attribute(node, [[1, conv_n_filter, in_ch]], [[1, 1, in_ch]])
                        if "None" not in conv_activation_fn:
                            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                                '{}_conv_activation_{}'.format(self.target, i),
                                conv_activation_fn, {})
                        update_node_attribute(node, [[1, 1, in_ch]], [[1, 1, in_ch]])
            else:
                node.update_operation(conv_op_type, {
                    'kernel_size': conv_kernel_size,
                    'in_channels': 1,
                    'out_channels': 1,
                    'padding': conv_kernel_size//2
                })
                update_node_attribute(node, [[1, 1, in_ch]], [[1, 1, in_ch]])
                if "None" not in conv_activation_fn:
                    node = graph.insert_node_on_edge(node.outgoing_edges[0],
                            '{}_conv_activation_0'.format(self.target),
                            conv_activation_fn, {})
                update_node_attribute(node, [[1, 1, in_ch]], [[1, 1, in_ch]])
            
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_pooling'.format(self.target),
                pooling_op_type, {
                'kernel_size': pooling_kernel_size,
                'stride': 1,
                'padding': pooling_kernel_size//2
            })
            update_node_attribute(node, [[1, 1, in_ch]], [[1, 1, in_ch]])
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_flatten'.format(self.target),
                "__torch__.nni.retiarii.nn.pytorch.Flatten", {})
            update_node_attribute(node, [[1, 1, in_ch]], [[1, in_ch]])
        else:
            node.update_operation("__torch__.nni.retiarii.nn.pytorch.Flatten", {})
            update_node_attribute(node, [[1, 1, in_ch]], [[1, in_ch]])

        # update the placeholder to be a new operation
        if fc_n_layer > 1:
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_fc_0'.format(self.target), fc_op_type, {
                    'in_features': in_ch,
                    'out_features': n_neurons
            })
            update_node_attribute(node, [[1, in_ch]], [[1, n_neurons]])
            if "None" not in fc_activation_fn:
                node = graph.insert_node_on_edge(node.outgoing_edges[0],
                    '{}_fc_activation_0'.format(self.target),
                    fc_activation_fn, {})
            update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
            if fc_n_layer > 2:
                for i in range(1, fc_n_layer-1):
                    node = graph.insert_node_on_edge(node.outgoing_edges[0],
                        '{}_fc_{}'.format(self.target, i),
                        fc_op_type, {
                            'in_features': n_neurons,
                            'out_features': n_neurons
                    })
                    update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
                    if "None" not in fc_activation_fn:
                        node = graph.insert_node_on_edge(node.outgoing_edges[0],
                            '{}_fc_activation_{}'.format(self.target, i),
                            fc_activation_fn, {})
                    update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])

            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_fc_{}'.format(self.target, fc_n_layer-1),
                fc_op_type, {
                    'in_features': n_neurons,
                    'out_features': 1
                })
            update_node_attribute(node, [[1, n_neurons]], [[1, 1]])
        else:
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_fc_0'.format(self.target), fc_op_type, {
                    'in_features': in_ch,
                    'out_features': 1
            })
            update_node_attribute(node, [[1, in_ch]], [[1, 1]])


@nni.trace
class MLPMutator(Mutator):
    def __init__(self, target):
        super(MLPMutator, self).__init__()
        self.target = target

    def mutate(self, model):
        nodes = model.get_nodes_by_label(self.target)
        assert len(nodes) == 1
        node = nodes[0]
        node.name = self.target + "_0"
        graph = node.graph
        
        related_info = node.operation.parameters

        # tail parameters
        n_input_features = related_info['n_input_features']
        n_neurons = self.choice(related_info['n_neurons_option'])
        fc_activation_fn = self.choice(related_info['fc_activation_fn_option'])
        fc_n_layer = self.choice(related_info['fc_n_layer_option'])
        dropout_rate = self.choice(related_info['dropout_option'])
        # update the placeholder to be a new operation
        if fc_n_layer > 1:
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_fc_0'.format(self.target), "__torch__.nni.retiarii.nn.pytorch.Linear", {
                    'in_features': n_input_features,
                    'out_features': n_neurons
            })
            update_node_attribute(node, [[1, n_input_features]], [[1, n_neurons]])
            if "None" not in fc_activation_fn:
                node = graph.insert_node_on_edge(node.outgoing_edges[0],
                    '{}_fc_activation_0'.format(self.target),
                    fc_activation_fn, {})
                update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
            if dropout_rate > 0:
                node = graph.insert_node_on_edge(node.outgoing_edges[0],
                        '{}_fc_dropout_0'.format(self.target),
                        "__torch__.nni.retiarii.nn.pytorch.Dropout", {'p': dropout_rate})
                update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
                
            if fc_n_layer > 2:
                for i in range(1, fc_n_layer-1):
                    node = graph.insert_node_on_edge(node.outgoing_edges[0],
                        '{}_fc_{}'.format(self.target, i), "__torch__.nni.retiarii.nn.pytorch.Linear", {
                            'in_features': n_neurons,
                            'out_features': n_neurons
                    })
                    update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
                    if "None" not in fc_activation_fn:
                        node = graph.insert_node_on_edge(node.outgoing_edges[0],
                            '{}_fc_activation_{}'.format(self.target, i),
                            fc_activation_fn, {})
                        update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
                    if dropout_rate > 0:
                        node = graph.insert_node_on_edge(node.outgoing_edges[0],
                                '{}_fc_dropout_{}'.format(self.target, i),
                                "__torch__.nni.retiarii.nn.pytorch.Dropout", {'p': dropout_rate})
                        update_node_attribute(node, [[1, n_neurons]], [[1, n_neurons]])
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_fc_{}'.format(self.target, fc_n_layer-1),
                "__torch__.nni.retiarii.nn.pytorch.Linear", {
                    'in_features': n_neurons,
                    'out_features': 1
                })
            update_node_attribute(node, [[1, n_neurons]], [[1, 1]])
        else:
            node = graph.insert_node_on_edge(node.outgoing_edges[0],
                '{}_fc_0'.format(self.target), "__torch__.nni.retiarii.nn.pytorch.Linear", {
                    'in_features': n_input_features,
                    'out_features': 1
            })
            update_node_attribute(node, [[1, n_input_features]], [[1, 1]])
        


# class ResNetMutator(Mutator):
#     def __init__(self, target):
#         super(ResNetMutator, self).__init__()
#         self.target = target

#     def mutate_for_resnet(self, node):
#         related_info = node.operation.parameters


#     def mutate(self, model):
#         nodes = model.get_nodes_by_label(self.target)
#         assert len(nodes) == 1
#         node = nodes[0]
#         node.name = self.target + "_0"
#         graph = node.graph
        