"""
This module returns the co-safe ltl tasks that we used in our experiments.
The set of propositional symbols are {a,b,c,d,e,f,g,h,n,s}:
    a: got_wood
    b: used_toolshed
    c: used_workbench
    d: got_grass
    e: used_factory
    f: got_iron
    g: used_bridge
    h: used_axe
    n: is_night
    s: at_shelter
"""


def get_sequence_of_subtasks():
    # Experiment 1: Sequences of Sub-Tasks (Section 5.2 in paper)
    tasks = [
        _get_sequence('ba'),
        _get_sequence('ac'),
        _get_sequence('de'),
        _get_sequence('db'),
        _get_sequence('fae'),
        _get_sequence('abdc'),
        _get_sequence('acfb'),
        _get_sequence('acfc'),
        _get_sequence('faeg'),
        _get_sequence('acfbh')
    ]
    return tasks


def get_sequence_of_subtasks2():
    # Experiment 1: Sequences of Sub-Tasks (Section 5.2 in paper)
    tasks = [
        _get_sequence('ba'),
        _get_sequence('ac'),
        _get_sequence('de'),
        _get_sequence('db'),
        _get_sequence('fae'),
        _get_sequence('abdc'),
        # _get_sequence('acfb'),
        # _get_sequence('acfc'),
        # _get_sequence('faeg'),
        # _get_sequence('acfbh')
    ]
    return tasks

def get_sequence_of_until():
    tasks = [
        _get_until('ba'),
        _get_until('ac'),
        _get_until('de'),
        _get_until('db'),
        _get_until('fae'),
        _get_until('abdc'),
    ]
    return tasks


def get_interleaving_subtasks():
    # Experiment 2: Interleaving Sub-Tasks (Section 5.3 in paper)
    tasks = [
        _get_sequence('ab'),
        _get_sequence('ac'),
        _get_sequence('de'),
        _get_sequence('db'),
        ('and', _get_sequence('ae'), _get_sequence('fe')),
        ('and', _get_sequence('dc'), _get_sequence('abc')),
        ('and', _get_sequence('fb'), _get_sequence('acb')),
        ('and', _get_sequence('fc'), _get_sequence('ac')),
        ('and', _get_sequence('aeg'), _get_sequence('feg')),
        ('and', _get_sequence('fbh'), _get_sequence('acbh'))
    ]
    return tasks


def get_safety_constraints():
    # Experiment 3: Safety Constraints (Section 5.4 in paper)
    tasks = [
        _get_sequence_night('ab'),
        _get_sequence_night('ac'),
        _get_sequence_night('de'),
        _get_sequence_night('db'),
        ('and', _get_sequence_night('ae'), _get_sequence_night('fe')),
        ('and', _get_sequence_night('dc'), _get_sequence_night('abc')),
        ('and', _get_sequence_night('fb'), _get_sequence_night('acb')),
        ('and', _get_sequence_night('fc'), _get_sequence_night('ac')),
        ('and', _get_sequence_night('aeg'), _get_sequence_night('feg')),
        ('and', _get_sequence_night('fbh'), _get_sequence_night('acbh'))
    ]
    return tasks


def get_option(goal):
    return _get_sequence(goal)


def get_option_night(goal):
    return _get_sequence_night(goal)


def _get_sequence(seq):
    if len(seq) == 1:
        return ('until', 'True', seq)
    return ('until', 'True', ('and', seq[0], _get_sequence(seq[1:])))


def _get_until(seq):
    if len(seq) == 1:
        return ('until', 'True', seq)
    return ('and', ('until', ('not', seq[1]), seq[0]), _get_until(seq[1:]))


def _sn():
    # returns formula to stay on the shelter
    return ('or', ('not', 'n'), 's')


def _snp(proposition):
    # adds the special constraint to go to the shelter for a proposition
    return ('or', ('and', ('not', 'n'), proposition), ('and', 's', proposition))


def _get_sequence_night(seq):
    if len(seq) == 1:
        return ('until', _sn(), _snp(seq))
    return ('until', _sn(), ('and', _snp(seq[0]), _get_sequence_night(seq[1:])))


def _not(seq):
    """
    negate only atomic propositions following co-safe syntax, e.g. !a: correct; !(a&b): incorrect
    """
    if isinstance(seq, str) and len(seq) == 1:
        return 'not', seq
    raise NotImplementedError("The following formula doesn't follow the cosafe syntactic restriction: " + str(seq))


def _until(seq1, seq2):
    return 'until', seq1, seq2


def _next(seq):
    return 'next', seq


def _get_sequence_generic(*seq):
    """
    e.g. 'seq' = 'a', ('next', ('until', 'True', 'b')) or 'abc'
    """
    print("seq: ", seq)
    if len(seq) == 1:
        print("seq[0]", seq[0])
        if len(seq[0]) == 1:
            if isinstance(seq[0], str):
                return 'until', 'True', seq[0]
            else:
                print("seq[0][0]", seq[0][0])
                return seq[0][0]
        else:
            if len(seq[0][0]) > 1:  # 1st element is an operator, e.g. seq=(('next', 'b'),)
                return 'until', 'True', seq[0]
            return 'until', 'True', ('and', seq[0][0], _get_sequence_generic(seq[0][1:]))
    else:
        print("seq[1:]: ", seq[1:])
        return 'until', 'True', ('and', seq[0], _get_sequence_generic(seq[1:]))


######### The following methods are for transfer learning #########
def get_sequence_training_tasks():
    """ Sequence training tasks for the transfer tasks. """
    tasks = [
        _get_sequence('ab'),
        _get_sequence('ac'),
        _get_sequence('de'),
        _get_sequence('db'),
        _get_sequence('fae'),
        _get_sequence('abdc'),
        _get_sequence('acfb'),
        _get_sequence('acfc'),
        _get_sequence('faeg'),
        _get_sequence('acfbh')
    ]
    return tasks


def get_interleaving_training_tasks():
    """ Interleaving training tasks for the transfer tasks. """
    tasks = [
        _get_sequence('ab'),
        _get_sequence('ac'),
        _get_sequence('de'),
        _get_sequence('db'),
        ('and', _get_sequence('ae'), _get_sequence('fe')),
        ('and', _get_sequence('dc'), _get_sequence('abc')),
        ('and', _get_sequence('fb'), _get_sequence('acb')),
        ('and', _get_sequence('fc'), _get_sequence('ac')),
        ('and', _get_sequence('aeg'), _get_sequence('feg')),
        ('and', _get_sequence('fbh'), _get_sequence('acbh'))
    ]
    return tasks


def get_transfer_tasks():
    """ Testing tasks for the transfer tasks. """
    tasks = [
        _get_sequence('ab'),
        _get_sequence('ac'),
        _get_sequence('de'),
        _get_sequence('db'),
        ('and', _get_sequence('ae'), _get_sequence('fe')),
        ('and', _get_sequence('dc'), _get_sequence('abc')),
        ('and', _get_sequence('fb'), _get_sequence('acb')),
        ('and', _get_sequence('fc'), _get_sequence('ac')),
        ('and', _get_sequence('aeg'), _get_sequence('feg')),
        ('and', _get_sequence('fbh'), _get_sequence('acbh')),

        ('and', _get_sequence('fbh'), _get_sequence('cbh')),
        _get_sequence('deg'),  # _get_sequence('de'), ('and', _get_sequence('aeg'), _get_sequence('feg'))
        _get_sequence('dcb'),  # ('and', _get_sequence('dc'), _get_sequence('abc')), ('and', _get_sequence('fb'), _get_sequence('acb'))
        _get_sequence('af'),  # a & !f is a subset of a & !e & f: ('and', _get_sequence('ae'), _get_sequence('fe'))

        _get_sequence('agc'),

        # _get_sequence_generic('ab'),  # F(a & Fb)
        # ('and', _until(_not('b'), 'a'), _get_sequence_generic('b')),  # !b U a & F(b)
        # _next(_get_sequence_generic('b')),  # XFb
        # _get_sequence_generic(_next('b')),  # FXb
        # _get_sequence_generic('a', _next(_get_sequence_generic('b'))),  # F(a & XFb)
        # _get_sequence_generic('a', _get_sequence_generic(_next('b'))),  # F(a & FXb)
        # _get_sequence_generic('a', _next('b')),  # F(a & Xb)
    ]
    return tasks
