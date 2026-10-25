"""Shared, stateless USI rewrite. No elapsed-time or 420-second bookkeeping."""
BUOY = ['5i5h', '5a5b', '5h5i', '5b5a']

def position_ply(command):
    t = command.split()
    if t[:2] == ['position', 'startpos']:
        n, turn = 1, 0
        offset = 2
    elif t[:2] == ['position', 'sfen'] and len(t) >= 6:
        n, turn = int(t[5]), int(t[3] == 'w')
        offset = 6
    else:
        raise ValueError('unsupported position')
    moves = t[offset+1:] if len(t) > offset and t[offset] == 'moves' else []
    return n + len(moves), turn ^ (len(moves) % 2)

def rewrite(command, position, rule='none', k=15, cap_ms=30000, byoyomi_ms=3000):
    if rule == 'none' or not position:
        return command
    ply, color = position_ply(position)
    own_index = (ply - 5) // 2 + 1
    # Exclude the four buoy moves. A ponder position includes the predicted reply.
    if color != 0 or not 1 <= own_index <= k:
        return command
    t = command.split()
    if not t or t[0] != 'go' or any(x in t for x in ['infinite', 'mate', 'nodes', 'depth', 'movetime', 'rtime']):
        return command
    if rule == 'cap':
        if 'btime' in t:
            i = t.index('btime') + 1
            t[i] = str(min(int(t[i]), cap_ms))
    elif rule == 'byoyomi':
        for key in ['btime', 'binc', 'winc', 'byoyomi']:
            if key in t:
                i = t.index(key)
                del t[i:i+2]
        t += ['btime', '0', 'binc', '0', 'winc', '0', 'byoyomi', str(byoyomi_ms)]
    else:
        raise ValueError(rule)
    return ' '.join(t)
