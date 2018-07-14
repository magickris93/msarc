def file_write_msf(alignment, f, infile=None, outfile="", prot=True, endgap=None):
    from datetime import datetime
    from Bio.SeqUtils.CheckSum import gcg
    from .sequence import MSF_SPACE, MSF_TERMINAL_SPACE, SPACES
    from .probA import ENDGAP_NONE

    length = alignment.get_alignment_length()
    idlen = 0
    lenlen = max(5, len(str(length)))
    remainder = length % 50
    blocks_last = (remainder - 1) // 10 + 1

    check = []
    seqs = []
    for record in alignment:
        idlen = max(idlen, len(record.id))
        seq = str(record.seq)
        check.append(gcg(seq))
        for space in SPACES:
            if space is MSF_SPACE:
                continue
            seq = seq.replace(space, MSF_SPACE)
        if endgap != ENDGAP_NONE:
            for i in xrange(length):
                if seq[i] is not MSF_SPACE:
                    break
            seq = MSF_TERMINAL_SPACE * i + seq[i:]
            for i in xrange(length - 1, -1, -1):
                if seq[i] is not MSF_SPACE:
                    break
            seq = seq[:i + 1] + MSF_TERMINAL_SPACE * (length - i - 1)
        seqs.append(seq)
    allcheck = sum(check) % 10000

    head_fmt = " {}  MSF: {}  Type: {}  {:%B %d, %Y %H:%M}  Check: {} ..\n\n"
    seq_fmt = " Name: {{:<{}}}  Len: {{:>{}}}  Check: {{:>4}}  Weight:  1.00\n".format(idlen + 14 - lenlen, lenlen)
    len_fmt = "{{:<{}}}  {{:<10}} {{:>43}}\n".format(idlen)
    aln_fmt = "{{:<{}}}  {{}} {{}} {{}} {{}} {{}}\n".format(idlen)
    aln_fmt_last = "{{:<{}}} {}\n".format(idlen, " {}" * blocks_last)
    if remainder >= lenlen * 2 + 1:
        len_fmt_last = "{{:<{}}}  {{:<{}}} {{:>{}}}\n".format(idlen, lenlen, remainder - lenlen + blocks_last - 2)
    else:
        len_fmt_last = "{{:<{}}}  {{}}\n".format(idlen)

    f.write("!!{}A_MULTIPLE_ALIGNMENT 1.0\n".format("A" if prot else "N"))
    if infile is None:
        f.write("PileUp\n\n")
    else:
        f.write("PileUp of: @{}\n\n".format(infile))
    f.write(head_fmt.format(outfile, length, "P" if prot else "N", datetime.now(), allcheck))
    for i, record in enumerate(alignment):
        f.write(seq_fmt.format(record.id, length, check[i]))
    f.write("\n//\n\n")

    for o in xrange(0, length, 50):
        e = o + 50
        if e > length:
            len_fmt = len_fmt_last
            aln_fmt = aln_fmt_last
        f.write(len_fmt.format("", o + 1, min(length, e)))
        for i, record in enumerate(alignment):
            seq = seqs[i][o:e]
            f.write(aln_fmt.format(record.id, seq[0:10], seq[10:20], seq[20:30], seq[30:40], seq[40:50]))
        if e < length:
            f.write("\n")


def file_read_msf(f):
    from string import maketrans
    from collections import OrderedDict
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from .sequence import SPACES, SPACE, GAPPED_ALPHABET
    from .alignment import tomsa

    seq = OrderedDict()
    header = True
    for line in f:
        if header:
            if line[:2] == "//":
                header = False
            continue
        l = line.strip().split()
        if not l or not l[0]:
            continue
        if len(l) <= 2:
            try:
                il = [int(_l) for _l in l]
            except ValueError:
                pass
            else:
                if il[0] % 50 == 1 and (len(l) == 1 or (il[1] > il[0] and il[1] - il[0] < 50)):
                    continue
        seq[l[0]] = seq.get(l[0], "") + "".join(l[1:])

    records = []
    table = maketrans("".join(SPACES), SPACE * len(SPACES))
    for id, s in seq.iteritems():
        records.append(SeqRecord(Seq(s.translate(table), GAPPED_ALPHABET), id=id))

    return tomsa(tuple(records))
