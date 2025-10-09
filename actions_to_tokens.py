import numpy as np

actions = ['rotate_right', 'rotate_left', 'move_forward', 'move_backward']

actions2 = [('move_forward', 'rotate_left'), ('move_backward', 'rotate_left'), ('move_backward', 'rotate_right'), ('move_forward', 'rotate_right')]

act_to_tok = {(act, ): tok for tok, act in enumerate(actions)}
act_to_tok.update({tuple(): len(actions)})
act_to_tok.update({act: tok + len(act_to_tok) for tok, act in enumerate(actions2)})
