import re

def parts(u):
    m = re.match(r'^[a-z]+://([^/?#]*)([^?#]*)(\?[^#]*)?(#.*)?', u.lower())
    if not m:
        return u.lower(), '', '', ''
    return m.group(1), m.group(2), (m.group(3) or ''), (m.group(4) or '')

def feats(u):
    h, p, q, f = parts(u)
    out = []
    h = h.split(':')[0]
    labs = h.split('.')
    out += ['H:' + h, 'R:' + '.'.join(labs[-2:]), 'T:' + labs[-1]]
    out += ['L:' + l for l in labs[:-1]]
    segs = [s for s in p.split('/') if s]
    out.append('D:%d' % min(len(segs), 8))
    for i, s in enumerate(segs[:2]):
        out.append('S%d:%s' % (i, s[:20]))
    for t in re.split(r'[^a-z0-9]+', p):
        if 2 <= len(t) <= 20: out.append('P:' + t)
    for t in re.split(r'[^a-z0-9]+', q):
        if 2 <= len(t) <= 20: out.append('Q:' + t)
    out.append('HASQ:%d' % (1 if q else 0))
    out.append('HASF:%d' % (1 if f else 0))
    out.append('LEN:%d' % min(len(u) // 25, 8))
    nd = sum(c.isdigit() for c in u) / max(len(u), 1)
    out.append('DIG:%d' % int(nd * 10))
    return out
