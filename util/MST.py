# This code is the modifier version of https://github.com/mlbright/edmonds/blob/master/edmonds/edmonds.py
import numpy as np
import collections
import torch

class Digraph:
    '''We represent directed graphs using a map of outgoing edges for each node.
    '''
    new_node_id = 111111
    def __init__(self, successors, score=None):
        '''Initialize this digraph using a successors map and a score function.
        successors: A map from source node ids to lists of target nodes that can
          be reached from each source node. For instance, {1: [2], 2: [1, 3],
          3: [1]} represents a directed graph with three nodes (1, 2, 3) and two
          cycles, (1 -> 2 -> 1) and (1 -> 2 -> 3 -> 1). Similarly, {1: [],
          2: [3], 3: []} represents a directed graph with three nodes and one
          edge that connects node 2 to node 3.
        score: A callable that takes two node ids and returns a scalar
          score for the directed edge between those two nodes. Defaults to a
          static function where all edges have score 0.
        '''
        self.successors = successors
        self.score = score

    def __contains__(self, x):
        '''Return True iff x is a node in our Digraph.'''
        return x in self.successors

    def __iter__(self):
        '''Iterate over the nodes in our graph.'''
        return iter(self.successors)

    def num_nodes(self):
        '''Return the number of nodes in this Digraph.'''
        return len(self.successors)

    def num_edges(self):
        '''Return the number of edges in this Digraph.'''
        return sum(1 for _ in self.iteredges())

    def dot(self, name):
        '''Get this graph as a dot string.'''
        nodes = ' '.join('_%s_%s;' % (x, name) for x in self)
        edges = ' '.join(
            '_%s_%s -> _%s_%s [label="%.2f"];' % (
                s, name, t, name, self.score(s, t))
            for s, t in self.iteredges())
        return 'digraph _%s {%s %s}' % (name, nodes, edges)

    def iteredges(self):
        '''Iterate over the pairs of node ids in all edges in this Digraph.'''
        for source, targets in self.successors.items():
            for target in targets:
                yield source, target

    def mst(self):
        '''Return the MST of this Digraph using the Chu-Liu-Edmonds algorithm.
        Returns a new Digraph.
        '''
        mark = Digraph.new_node_id
        candidate = self.greedy()
        cycle = candidate.find_cycle()
        if not cycle:
            return candidate
        new_id, old_edges, compact = self.contract(cycle)
        merged = self.merge(compact.mst(), new_id, old_edges, cycle)
        return merged

    def find_cycle(self):
        '''Find and return a cycle in our Digraph, or None.'''
        # from guido's blog :
        # http://neopythonic.blogspot.com/2009/01/detecting-cycles-in-directed-graph.html
        worklist = set(self.successors)
        while worklist:
            stack = [worklist.pop()]
            while stack:
                top = stack[-1]
                for node in self.successors.get(top, ()):
                    try:
                        # raises ValueError if node is not in stack.
                        cycle = stack[stack.index(node):]
                        succs = dict((source, [cycle[(i + 1) % len(cycle)]])
                                     for i, source in enumerate(cycle))
                        return Digraph(succs, self.score)
                    except ValueError:
                        pass
                    if node in worklist:
                        stack.append(node)
                        worklist.remove(node)
                        break
                else:
                    stack.pop()
        return None

    def contract(self, cycle):
        '''Given a cycle in our graph, contract it into a single node.
        Returns a tuple (id, graph). The graph is a new Digraph instance
        containing no nodes from the cycle, with one extra new node created to
        represent the cycle. The id is the id of the new node.
        '''
        # create a new id to represent the cycle in the resulting graph.
        new_id = Digraph.new_node_id
        Digraph.new_node_id += 1

        # we store links that cross into and out of the cycle in these maps. the
        # to_cycle map contains links reaching into the cycle, and is thus a map
        # from each target node in the cycle to a list of source nodes that
        # reach that target from outside the cycle. the from_cycle map contains
        # links going out from the cycle, and is thus a map from each source
        # node in the cycle to a list of target nodes outside the cycle.
        to_cycle = collections.defaultdict(list)
        from_cycle = collections.defaultdict(list)

        scores = {}
        succs = collections.defaultdict(list)
        for source, target in self.iteredges():
            if source in cycle:
                if target not in cycle:
                    from_cycle[target].append(source)
            elif target in cycle:
                # we know source is not in cycle from above.
                to_cycle[source].append(target)
            else:
                succs[source].append(target)
                succs[target]
                scores[source, target] = self.score[source, target]

        old_edges = collections.defaultdict(list)

        # for each target in our graph that's reachable from the cycle, add an
        # edge from our new node to that target, with an appropriate score.
        for target, sources in from_cycle.items():
            succs[new_id].append(target)
            max_score = -1e100
            max_source = None
            for s in sources:
                score = self.score[s, target]
                if score > max_score:
                    max_score = score
                    max_source = s
            old_edges[max_source].append(target)
            scores[new_id, target] = max_score

        # before we handle the to_cycle map, we need to build some convenience
        # information for the cycle -- total score, and predecessor edges.
        pred = {}
        cycle_score = 0
        for s, t in cycle.iteredges():
            pred[t] = s
            cycle_score += self.score[s, t]

        # for each source in our graph that reaches into the cycle, add an edge
        # from the source to our new node, with an appropriate edge score.
        for source, targets in to_cycle.items():
            succs[source].append(new_id)
            max_score = -1e100
            max_target = None
            for t in targets:
                score = self.score[source, t] - self.score[pred[t], t]
                if score > max_score:
                    max_score = score
                    max_target = t
            old_edges[source].append(max_target)
            scores[source, new_id] = cycle_score + max_score

        return new_id, old_edges, Digraph(succs, scores)

    def merge(self, mst, new_id, old_edges, cycle):
        '''Merge the nodes in an MST that were contracted from a cycle.
        We want to merge the information from the mst and the cycle into our
        graph to yield a subset of our original edges, using only edges from the
        MST, from the cycle, or from old_edges (the edges that were used when
        collapsing the cycle into the node with new_id in the MST).
        mst: A Digraph containing an MST of our graph, with a single node
          representing the nodes and edges from the cycle.
        new_id: The id of the collapsed node in MST that represents the cycle.
        old_edges: A dictionary mapping source to target for edges that were
          used to collapse the cycle. These are used to reconstruct the
          original edges from the graph.
        cycle: A Digraph containing nodes and edges in a cycle of our graph.
        Return a new Digraph containing the merged nodes and edges.
        '''
        succs = dict((n, []) for n in self)
        for source, target in mst.iteredges():
            if source == new_id:
                # this edge points out of the cycle into the mst. there might be
                # more than one of these. use the old_edges to find out which
                # cycle node is responsible for this edge, and add it.
                for s, ts in old_edges.items():
                    for t in ts:
                        if t == target:
                            succs[s].append(t)

            elif target == new_id:
                # this edge points at the cycle. use the old_edges to find out
                # where in the cycle it points, then add all the edges in the
                # cycle except the one that completes the loop.
                targets = old_edges[source]
                assert len(targets) == 1, targets
                target = targets[0]
                succs[source].append(target)
                cycle_source = target
                cycle_target = cycle.successors[cycle_source][0]
                while cycle_target != target:
                    succs[cycle_source].append(cycle_target)
                    cycle_source = cycle_target
                    cycle_target = cycle.successors[cycle_source][0]

            else:
                # this edge is completely in the mst, so add it and move on.
                succs[source].append(target)

        return Digraph(succs, self.score)

    def greedy(self):
        '''Return a Digraph consisting of the max scoring edge for each node.'''
        # for each node, find the incoming link with the highest score.
        max_scores = {}
        max_sources = {}
        for source, target in self.iteredges():
            score = self.score[source, target]
            max_score = max_scores.get(target)
            if max_score is None or score > max_score:
                max_scores[target] = score
                max_sources[target] = source
        # then build a graph out of just these links.
        succs = dict((n, []) for n in self)
        for target, source in max_sources.items():
            succs[source].append(target)
        return Digraph(succs, self.score)



def decode_mst(ys_link, t_len):
    # ys: [(batchsize*max_n_spans, max_n_spans+1), (all_spans, type_classes)]
    # ts: (batchsize, n_spans, gold_target)
    batchsize, max_n_spans, _ = ys_link.shape

    ys_mst = []
    for i, (y_matrix, l) in enumerate(zip(ys_link, t_len)):
        data, score = {}, {}
        for row, y_scores in enumerate(y_matrix[:l]):
            for col, y_score in enumerate(y_scores[:l]):
                # reverse the direction of edges
                try:
                    data[col].append(row)
                except:
                    data[col] = [row]

                score[(col, row)] = y_score.cpu().item()

        for row, y_score in enumerate(y_matrix[:l,max_n_spans]):
            try:
                data[max_n_spans].append(row)
            except:
                data[max_n_spans] = [row]
            score[(max_n_spans, row)] = y_score.cpu().item()

        y_tree = [-1 for _ in range(max_n_spans)]
        count = 0
        for _ in Digraph(data, score).mst().iteredges():
            y_tree[_[1]] = _[0]
            count += 1
        y_max = y_matrix.max(-1)[1].cpu()
        """
        if( 0 in (torch.tensor(y_tree[:l]) == y_max[:l])):
            print(y_tree)
            print(y_max)
            print(y_matrix)
        """
        assert(count == l), 'count should same as l({} {})'.format(count, l)
        ys_mst.append(y_tree)


    return torch.tensor(ys_mst, dtype=torch.long)
