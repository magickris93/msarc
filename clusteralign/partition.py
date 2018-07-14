from numpy import ndarray, array, empty, empty_like, zeros, ones, arange, uint8
from ._partition import bestmove, calcgains_group, fixgains_group


class GraphPartitioner(object):
    def __init__(self, graph):
        from .graph import Graph
        assert isinstance(graph, Graph), 'Parameter "graph" is not an instance of Graph'
        self._graph = graph

        self.__indices = None
        self.__gains = None
        self.__balance_weight = None
        self.__minbalance = None
        self.__weights_cumsum = None
        self.__edges_cumsum = None

    def partition(self, parts):
        if not isinstance(parts, ndarray):
            parts = array(parts)
        self._parts = parts

        precision = 1e-13 * self._numgroups
        prev_parts = []
        loop_test = True
        loop_count = 0
        while loop_test:
            best_parts, best_gain, best_balance = self._testmoves()
            if best_parts is None:
                break
            loop_count += 1
            if loop_count > 10 and best_gain < precision:
                for pb, pp in prev_parts:
                    if (pp == best_parts).all():
                        best_balance, idx = sorted((b, i) for i, (b, p) in enumerate(prev_parts))[0]
                        best_parts = prev_parts[idx][1]
                        loop_test = False
                        break
                else:
                    prev_parts.append((best_balance, best_parts))
            self._move([(i, best_parts[i]) for i in xrange(self._numgroups)])

        return self._parts

    @property
    def _numgroups(self):
        return self._graph.num_sequences

    @property
    def _lengths(self):
        return self._graph.lengths

    @property
    def _weights(self):
        return self._graph._weights

    def __cumsum_edges(self, edges):
        ce = empty_like(edges)
        for i, eti in enumerate(edges):
            ce[i] = eti.cumsum()
        return ce

    @property
    def _weights_cumsum(self):
        if self.__weights_cumsum is None:
            self.__weights_cumsum = self.__cumsum_edges(self._weights)
        return self.__weights_cumsum

    @property
    def _edges(self):
        return self._graph._pair_pp

    @property
    def _balance_weight(self):
        if self.__balance_weight is None:
            b = [0, 0]
            for i, w in enumerate(self._weights):
                b[0] += w[:self._parts[i]].sum()
                b[1] += w[self._parts[i]:].sum()
            self.__balance_weight = b
        return self.__balance_weight

    @property
    def _minbalance(self):
        if self.__minbalance is None:
            half_weight = sum(self._balance_weight) / 2
            non_empty_seqs = (self._lengths > 0).sum()
            self.__minbalance = max(half_weight - non_empty_seqs / 2, 1)
        return self.__minbalance

    @property
    def _gains(self):
        if self.__gains is None:
            self.__gains = self._calcgains()
        return self.__gains

    @property
    def _indices(self):
        if self.__indices is None:
            self.__indices = empty(self._numgroups, dtype=object)
            for i in xrange(self._numgroups):
                self.__indices[i] = arange(self._lengths[i] + 1)
        return self.__indices

    def _new_allowed(self, parts):
        allowed = empty(self._numgroups, dtype=object)
        for i in xrange(self._numgroups):
            allowed[i] = ones(self._lengths[i] + 1, dtype=uint8)
            allowed[i][parts[i]] = 0
        return allowed

    def _testmoves(self):
        parts = self._parts.copy()

        gain = 0.
        bestgain = None if min(self._balance_weight) < self._minbalance else gain
        bestparts = parts
        bestbalance = abs(self._balance_weight[0] - self._balance_weight[1])

        allowed = self._new_allowed(parts)
        while True:
            if self._balance_weight[0] < self._minbalance:
                direction = 1
            elif self._balance_weight[1] < self._minbalance:
                direction = -1
            else:
                direction = 0

            move = bestmove(self._parts, self._gains, allowed, direction)
            if move is None:
                break
            group, partition, delta_gain = move
            gain += delta_gain

            self._movepartition(group, partition)
            allowed[group][partition] = 0

            if min(self._balance_weight) < self._minbalance:
                continue
            balance = abs(self._balance_weight[0] - self._balance_weight[1])
            if bestgain is None or gain > bestgain:
                    bestparts = self._parts.copy()
                    bestgain = gain
                    bestbalance = balance

        if bestparts is parts:
            self._parts = parts
            return None, None, None
        return bestparts, bestgain, bestbalance

    def _move(self, steps):
        for partition in steps:
            self._movepartition(*partition)

    def _movepartition(self, group, partition):
        if partition == self._parts[group]:
            #no change
            return

        if self.__gains is not None:
            self._fixgains(group, partition)

        p_group = self._parts[group]
        max_partition, min_partition = max(partition, p_group), min(partition, p_group)
        dweights = self._weights_cumsum[group][max_partition - 1]
        if min_partition > 0:
            dweights -= self._weights_cumsum[group][min_partition - 1]

        side = self._getside(group, partition)
        other = 1 if side == 0 else 0
        self._balance_weight[side] -= dweights
        self._balance_weight[other] += dweights

        self._parts[group] = partition

    def _getside(self, group, partition):
        p = self._parts[group]
        if partition == p:
            raise ValueError
        elif partition > p:
            return 1
        else: # elif partition < p:
            return 0

    def _getnode(self, group, partition):
        p = self._parts[group]
        if partition == p:
            raise ValueError
        elif partition > p:
            return partition - 1
        else: # elif partition < p:
            return partition

    def _calcgain(self, group, partition):
        if partition == 0:
            return 0.

        node = partition - 1
        gain = 0.

        for i in xrange(self._numgroups):
            if i == group:
                continue

            p_i = self._parts[i]
            l_i = self._lengths[i]

            # symbol to symbol edges
            if group < i:           # symbol to symbol edge matrix
                edges = self._edges[group, i]
            else:                   # transposed symbol to symbol edge matrix
                edges = self._edges[i, group].transpose()
            if p_i > 0:
                gain += edges[node, :p_i].sum()
            if p_i < l_i:
                gain -= edges[node, p_i:].sum()

        return gain

    def _calcgain_all(self, group):
        l_group = self._lengths[group]
        gains = zeros(l_group + 1, dtype=float)
        if not l_group:
            return gains

        for i in xrange(self._numgroups):
            if i == group:
                continue

            p_i = self._parts[i]    # the current point of partition
            l_i = self._lengths[i]

            if group < i:           # edge matrix
                edges = self._edges[group, i].toarray()
            else:                   # transposed edge matrix
                edges = self._edges[i, group].transpose().toarray()
            if p_i > 0:
                gains[1:] += edges[:, :p_i].sum(axis=1)
            if p_i < l_i:
                gains[1:] -= edges[:, p_i:].sum(axis=1)

        # divide by 2 here so we don't have to double the difference in
        # _fixgain_all, which is called more frequently
        return gains / 2.

    def _calcgains(self):
        g = empty(self._numgroups, dtype=object)
        for i in xrange(self._numgroups):
            g[i] = calcgains_group(self, i)
        return g

    def _fixgain(self, group, partition, move_group, move_partition):
        if (move_group == group
            or partition == 0
            or move_partition == self._parts[move_group]):
            return 0.

        node = partition - 1
        gain = 0.

        p_move_group = self._parts[move_group]

        # symbol to symbol edges
        if group < move_group:      # symbol to symbol edge matrix
            edges = self._edges[group, move_group]
        else:                       # transposed symbol to symbol edge matrix
            edges = self._edges[move_group, group].transpose()
        gain += edges[node, move_partition] * 2

        if move_partition < p_move_group:
            gain = -gain

        return gain

    def _fixgain_all(self, group, move_group, move_partition):
        if move_group == group:
            raise ValueError
        l_group = self._lengths[group]
        p_move_group = self._parts[move_group]
        gains = zeros(l_group + 1, dtype=float)
        if not l_group or move_partition == p_move_group:
            return gains

        if group < move_group:      # edge matrix
            edges = self._edges[group, move_group].toarray()
        else:                       # transposed edge matrix
            edges = self._edges[move_group, group].transpose().toarray()
        if move_partition > p_move_group:
            gains[1:] += edges[:, p_move_group:move_partition].sum(axis=1)
        else:   # less
            gains[1:] -= edges[:, move_partition:p_move_group].sum(axis=1)

        return gains

    def _fixgains(self, group, partition):
        p_group = self._parts[group]
        if partition == p_group:    #no change
            return

        g = self._gains
        for i in xrange(self._numgroups):
            if i == group or not self._lengths[i]:
                continue
            fixgains_group(self, i, group, partition, g[i])
