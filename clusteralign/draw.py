from math import floor
from collections import defaultdict
from .sequence import SPACES
from .alignment import Alignment


COLOR_DEFAULT_S = 1
COLOR_DEFAULT_V = 1


def hsv2rgb(h, s=COLOR_DEFAULT_S, v=COLOR_DEFAULT_V):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def sorttree(tree, order):
    for subclade in tree.root.clades:
        sorttree(subclade, order)
    def sorter(c):
        terminals = c.get_terminals()
        return sum(order[t.name] for t in terminals) / float(len(terminals))
    tree.root.clades.sort(key=sorter, reverse=False)


def colortree(tree, root=None, map=None, terminals=None, grayscale=True):
    root = tree
    map = {}
    terminals = []

    def _distance(node1, node2):
        return root.distance(node1, node2) ** 2

    def _colortree(tree):
        if tree.root.is_terminal():
            if not terminals:
                map[tree.root] = 0.
            else:
                clade = terminals[-1]
                map[tree.root] = map[clade] + _distance(clade, tree.root)
            terminals.append(tree.root)
            return
        color_sum = 0.
        branch_sum = 0.
        for subclade in tree.root.clades:
            _colortree(subclade)
            color_sum += map[subclade] * subclade.branch_length
            branch_sum += subclade.branch_length
        map[tree.root] = color_sum / branch_sum

    _colortree(tree)

    max_color = max(map.itervalues())
    complete_max_color = max_color + _distance(terminals[0], terminals[-1])
    for clade, color in map.iteritems():
        _color = (max_color - color) / complete_max_color
        clade._color_value = _color
        if grayscale:
            clade.color = hsv2rgb(0., 0., _color)
        else:
            clade.color = hsv2rgb(_color * 360.)


def normalized_column_probability(d):
    l = d.nonzero
    if l > 1:
        dp = d.value /  (l * (l - 1) / 2)
    else:
        dp = 0.0
    return dp


def draw(alignment, filename, height = 3, width = 1, size = 8, symbols=True, grayscale=True, sequence_colors=None):
    if type(alignment) is Alignment:
        alignment.alignment
        alignment.value
        length = len(alignment._columns)
        records = len(alignment._records)
        columns = alignment._columns
        cuts = alignment._cuts
        cut_levels = alignment._cut_levels
        max_level = float(max(alignment._cut_levels.values()))
        lengths = alignment._graph.lengths
        alignment = alignment.alignment
    else:
        length = len(alignment[0])
        records = len(alignment)
        columns = None
        cuts = None
        cut_levels = None
        max_level = None
        lengths = []
        for r in alignment:
            l = length
            for s in SPACES:
                l -= str(r.seq).count(s)
            lengths.append(l)

    f = open(filename, "w")

    print >>f, "%!PS-Adobe-3.0 EPSF-3.0"
    print >>f, "%%BoundingBox:", 0, 0, (length - 1) * (size * width) + size, (records - 1) * (size * height) + size
    print >>f, """
/LINE {
   newpath
   moveto
   lineto
   setgray
   stroke
} bind def
/BOX {
   newpath
   moveto
   0 8 rlineto
   8 0 rlineto
   0 -8 rlineto
   -8 0 rlineto
   closepath
   setgray
   fill
} bind def
/CBOX {
   newpath
   moveto
   0 8 rlineto
   8 0 rlineto
   0 -8 rlineto
   -8 0 rlineto
   closepath
   %f %f sethsbcolor
   fill
} bind def
/CHAR {
   newpath
   moveto
   setgray
   show
} bind def
/Consolas findfont""" % (COLOR_DEFAULT_S, COLOR_DEFAULT_V)
    print >>f, size * 0.95, "scalefont setfont"

    positions_l = defaultdict(dict)
    positions_r = defaultdict(dict)

    def char(c):
        return c

    def pos(x, y):
        return x * (size * width), (records - y - 1) * (size * height)

    def _lposrange(x, y):
        try:
            m1 = positions_l[y][x]
        except KeyError:
            m1 = None
        try:
            m2 = positions_r[y][x]
        except KeyError:
            m2 = None
        if m1 is None:
            m1 = m2
        if m2 is None:
            m2 = m1
        return min(m1, m2), max(m1, m2)

    def _lpos(x, y, _l, _r):
        l, r = _lposrange(x, y)
        if x == 0:
            return min((_l + _r) / 2., l)
        elif x == lengths[y]:
            return max((_l + _r) / 2., r)
        elif l >= _r <= r:
            return l
        elif l <= _l >= r:
            return r
        elif _l == _r:
            return _l
        else:
            return (l + r) / 2.

    def lpos(x, y, l, r):
        return pos(_lpos(x, y, l, r), y)

    def line(_x1, _y1, _x2, _y2, lv, l, r):
        x1, y1 = lpos(_x1, _y1, l, r)
        x2, y2 = lpos(_x2, _y2, l, r)
        print >>f, "%partition:", (x1, y1), (x2, y2)
        color = 1. - ((2 ** (max_level - lv)) / (2 ** max_level))
        color *= .9
        print >>f, color, x1, y1 + size, x2, y2, "LINE"
        print >>f, color, x1, y1 + size, x1, y1, "LINE"
        print >>f, color, x2, y2 + size, x2, y2, "LINE"

    def box(_x, _y, c, cp):
        x, y = pos(_x, _y)
        if grayscale:
            print >>f, cp, x, y, "BOX"
            if symbols:
                print >>f, "(%s)" % char(c), 0 if cp >= 0.5 else 1, x + (size * 0.2), y + (size * 0.2), "CHAR"
        else:
            print >>f, cp, x, y, "CBOX"
            if symbols:
                print >>f, "(%s)" % char(c), 0, x + (size * 0.2), y + (size * 0.2), "CHAR"

    for x in xrange(length):
        column = "".join(r[x] for r in alignment)
        print >>f, "%column:", column
        if columns:
            graph = columns[x]
            cp = 1. - normalized_column_probability(graph)
        else:
            graph = None
            cp = 0.
        for y, c in enumerate(column):
            if sequence_colors is not None:
                cp = sequence_colors[y]
            if c not in SPACES:
                if graph is not None:
                    positions_l[y][graph.starts[y]] = x
                    positions_r[y][graph.ends[y]] = x + 1
                box(x, y, c, cp)

    if cuts is not None:
        ordered_cuts = sorted((cut_levels[id], tuple(cut)) for cut, id in cuts)
        for level, cut in reversed(ordered_cuts):
            print >>f, "%partition:", cut
            l, r = 0, length
            for i in range(len(cut)):
                if 0 < cut[i] < lengths[i]:
                    l2, r2 = _lposrange(cut[i], i)
                    l, r = max(l, l2), min(r, r2)
            for i in range(len(cut) - 1):
                line(cut[i + 1], i + 1, cut[i], i, level, l, r)

    f.close()
